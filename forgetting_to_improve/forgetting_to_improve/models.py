import torch
import gpytorch
import numpy as np
import warnings
from botorch.utils.transforms import normalize
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Log
from botorch.sampling import IIDNormalSampler
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from linear_operator.operators import DiagLinearOperator


MIN_INFERRED_NOISE_LEVEL = 1e-4


def standardize(X_to_standardize, X):
    """Standardize data based on training data statistics."""
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X_to_standardize - X.mean(dim=0)) / X_std


def unstandardize(X_standardized, X):
    """Reverse standardization."""
    X_std = X.std(dim=0)
    X_std = X_std.where(X_std >= 1e-9, torch.full_like(X_std, 1.0))
    return (X_standardized * X_std) + X.mean(dim=0)


class ExactGPModel(gpytorch.models.ExactGP):
    """
    Simple exact GP model for the mean function.
    Stores training inputs for use by heteroskedastic likelihood.
    """
    
    def __init__(self, train_x, train_y, likelihood, covar_module):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module
        # Store training inputs for likelihood access
        self.train_inputs = (train_x,)
    
    def transform_inputs(self, X):
        """Identity transform for BoTorch compatibility."""
        return X
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class HeteroskedasticNoise(gpytorch.Module):
    """
    Heteroskedastic noise model that predicts noise as a function of inputs.
    Uses a GP to model log-transformed noise variance.
    """
    
    def __init__(self, noise_model, train_x):
        super().__init__()
        self.noise_model = noise_model
        self.train_x = train_x
    
    def forward(self, *params, shape=None, **kwargs):
        """Predict noise variance at training input locations."""
        # During training, use stored training inputs
        # Get log-variance predictions from noise model
        with torch.no_grad():
            noise_posterior = self.noise_model.posterior(self.train_x)
            # Exp to get variance (model predicts log-variance)
            noise_var = noise_posterior.mean.exp().clamp(min=MIN_INFERRED_NOISE_LEVEL)
        
        # Return as diagonal operator
        if noise_var.ndim > 1:
            noise_var = noise_var.squeeze(-1)
        
        return DiagLinearOperator(noise_var)


class HeteroskedasticGaussianLikelihood(gpytorch.likelihoods._GaussianLikelihoodBase):
    """
    Gaussian likelihood with heteroskedastic noise predicted by a noise model.
    """
    
    def __init__(self, noise_model, train_x):
        super().__init__(noise_covar=HeteroskedasticNoise(noise_model, train_x))


class HeteroscedasticGPModel:
    """
    Most Likely Heteroscedastic GP implementation based on BoTorch's HeteroskedasticSingleTaskGP.
    
    This implementation follows "Most Likely Heteroscedastic Gaussian Process Regression":
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf
    
    Key difference from FixedNoiseGP: This model can predict input-dependent noise variance
    at test points by using a secondary GP to model log-variance as a function of inputs.
    
    Implementation details:
    - Uses two GPs: one for mean function, one for log-variance function
    - Iteratively refines variance estimates until convergence
    - The noise GP uses Log() outcome transform to model log-variance
    - Can extrapolate noise predictions to unseen test points
    """
    
    def __init__(self, train_x, train_y, kernel, max_iter=10, tol=1e-04,
                 var_estimate='paper', var_samples=1000, norm_and_std=True):
        """
        Args:
            train_x: Training inputs (n, d)
            train_y: Training targets (n,) or (n, 1)
            kernel: Kernel (covar_module) for the mean GP
            max_iter: Maximum iterations for convergence
            tol: Convergence tolerance
            var_estimate: 'paper' for sampling-based variance estimation, 
                         'mcr' for mean of squared residuals
            var_samples: Number of samples for variance estimation (if var_estimate='paper')
            norm_and_std: Whether to normalize inputs and standardize outputs
        """
        self.train_x_original = train_x
        self.train_y_original = train_y
        self.norm_and_std = norm_and_std
        self.max_iter = max_iter
        self.tol = tol
        self.var_estimate = var_estimate
        self.var_samples = var_samples
        
        # Ensure train_y is 1D for internal use
        if train_y.ndim > 1:
            train_y = train_y.squeeze(-1)
        
        # Store normalization/standardization parameters
        if norm_and_std:
            self.X_bounds = torch.stack([train_x.min() * torch.ones(train_x.shape[1]),
                                         train_x.max() * torch.ones(train_x.shape[1])])
            self.train_x = normalize(train_x, self.X_bounds)
            self.train_y = standardize(train_y, train_y)
            self.y_mean = train_y.mean()
            self.y_std = train_y.std()
            # Ensure std is not too small
            if self.y_std < 1e-9:
                self.y_std = torch.tensor(1.0)
        else:
            self.train_x = train_x.clone()
            self.train_y = train_y.clone()
            self.X_bounds = None
            self.y_mean = torch.tensor(0.0)
            self.y_std = torch.tensor(1.0)
        
        self.covar_module = kernel
        
        # BoTorch compatibility attributes
        self.num_outputs = 1
        self._num_outputs = 1
        
        # Will be set during fitting
        self.mll = None
        self.model = None
        self.noise_model = None
        self.noise_log_var_mean = torch.tensor(0.0)
        self.noise_log_var_std = torch.tensor(1.0)
    
    def fit(self):
        """
        Fit the heteroscedastic GP using the Most Likely approach from the paper.
        
        This implements Algorithm 1 from the paper:
        1. Fit a homoscedastic GP
        2. Estimate observation variances from the predictive posterior
        3. Fit a heteroscedastic GP with the estimated variances (including a noise GP)
        4. Update variance estimates
        5. Repeat until convergence
        """
        # Step 1: Fit initial homoscedastic model
        homo_model = SingleTaskGP(
            train_X=self.train_x, 
            train_Y=self.train_y.unsqueeze(-1) if self.train_y.ndim == 1 else self.train_y,
            covar_module=self.covar_module
        )
        homo_model.likelihood.noise_covar.register_constraint("raw_noise",
                                                              GreaterThan(1e-5))
        homo_mll = gpytorch.mlls.ExactMarginalLogLikelihood(homo_model.likelihood,
                                                            homo_model)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*scipy_minimize.*")
            warnings.filterwarnings("ignore", ".*OptimizationWarning.*")
            fit_gpytorch_mll(homo_mll)
        
        # Get initial posterior
        homo_mll.eval()
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
            homo_posterior = homo_mll.model.posterior(self.train_x)
        
        # Step 2: Estimate initial observed variance
        if self.var_estimate == 'mcr':
            # Mean of Conditional Residuals
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
                observed_var = torch.tensor(
                    np.power(homo_mll.model.posterior(self.train_x).mean.numpy().reshape(-1,) - 
                            self.train_y.numpy(), 2),
                    dtype=torch.float
                )
        else:
            # Sampling-based variance estimation (from the paper)
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
                sampler = IIDNormalSampler(sample_shape=torch.Size([self.var_samples]))
                predictive_posterior = homo_mll.model.posterior(self.train_x, observation_noise=True)
                samples = sampler(predictive_posterior)
                # samples shape: (var_samples, n, 1), we want (n,)
                observed_var = 0.5 * ((samples - self.train_y.unsqueeze(0).unsqueeze(-1))**2).mean(dim=0).squeeze(-1)
        
        # Clamp variance to reasonable range for numerical stability
        observed_var = torch.clamp(observed_var, min=1e-6, max=10.0)
        
        old_mean = homo_posterior.mean
        old_variance = homo_posterior.variance
        saved_hetero_model = None
        saved_noise_model = None
        
        # Step 3: Iterative refinement with noise model
        for i in range(self.max_iter):            
            # Create noise model (GP that predicts log-variance)
            # Use a simple likelihood without aggressive priors for robustness
            noise_likelihood = GaussianLikelihood(
                noise_constraint=GreaterThan(1e-4, transform=None, initial_value=1e-3)
            )
            
            # Manually take log of observed_var for training
            # Standardize log variance for better numerical stability
            log_observed_var = torch.log(observed_var)
            log_var_mean = log_observed_var.mean()
            log_var_std = log_observed_var.std()
            if log_var_std < 1e-6:
                log_var_std = torch.tensor(1.0)
            log_observed_var_std = (log_observed_var - log_var_mean) / log_var_std
            
            noise_model = SingleTaskGP(
                train_X=self.train_x,
                train_Y=log_observed_var_std.unsqueeze(-1),
                likelihood=noise_likelihood,
            )
            
            # Fit noise model with robust options
            noise_mll = gpytorch.mlls.ExactMarginalLogLikelihood(noise_model.likelihood, noise_model)
            
            # Try fitting with standard options first, then with more robust options if it fails
            noise_fit_success = False
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*scipy_minimize.*")
                    warnings.filterwarnings("ignore", ".*OptimizationWarning.*")
                    fit_gpytorch_mll(noise_mll)
            except:
                # Try with looser options
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        from botorch.optim.fit import fit_gpytorch_mll_scipy
                        fit_gpytorch_mll_scipy(noise_mll, options={"maxiter": 50})
                except:
                    pass
            
            # Store standardization params for later use
            saved_log_var_mean = log_var_mean
            saved_log_var_std = log_var_std
            
            # Create heteroskedastic likelihood with the noise model
            hetero_likelihood = HeteroskedasticGaussianLikelihood(noise_model, self.train_x)
            
            # Create mean model with heteroskedastic likelihood
            hetero_model = ExactGPModel(
                self.train_x,
                self.train_y,
                hetero_likelihood,
                self.covar_module
            )
            
            # Fit mean model with robust options
            hetero_mll = gpytorch.mlls.ExactMarginalLogLikelihood(hetero_likelihood, hetero_model)
            
            mean_fit_success = False
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*scipy_minimize.*")
                    warnings.filterwarnings("ignore", ".*OptimizationWarning.*")
                    fit_gpytorch_mll(hetero_mll)
                mean_fit_success = True
            except:
                # Try with looser options
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        from botorch.optim.fit import fit_gpytorch_mll_scipy
                        fit_gpytorch_mll_scipy(hetero_mll, options={"maxiter": 50})
                    mean_fit_success = True
                except:
                    pass
            
            if not mean_fit_success:
                # If mean model fitting fails, return previous model
                if saved_hetero_model is not None and saved_noise_model is not None:
                    self.model = saved_hetero_model
                    self.noise_model = saved_noise_model
                    self.noise_log_var_mean = saved_log_var_mean
                    self.noise_log_var_std = saved_log_var_std
                    return
                else:
                    # On first iteration, use the model even if fitting failed
                    pass
            
            hetero_model.eval()
            noise_model.eval()
            
            with torch.no_grad(), warnings.catch_warnings():
                warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
                hetero_posterior = hetero_model(self.train_x)
            
            new_mean = hetero_posterior.mean.unsqueeze(-1)
            new_variance = hetero_posterior.variance.unsqueeze(-1)
            
            # Check convergence
            means_eq = torch.all(torch.lt(torch.abs(torch.add(old_mean, -new_mean)), self.tol))
            
            variances_eq = torch.all(torch.lt(torch.abs(torch.add(old_variance, -new_variance)), self.tol))
            
            if means_eq and variances_eq:
                self.model = hetero_model
                self.noise_model = noise_model
                return
            
            saved_hetero_model = hetero_model
            saved_noise_model = noise_model
            old_mean = new_mean
            old_variance = new_variance
            
            # Update variance estimates
            if self.var_estimate == 'mcr':
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
                    mean_pred = hetero_model(self.train_x).mean
                    observed_var = torch.pow(mean_pred - self.train_y, 2)
            else:
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.filterwarnings("ignore", ".*The input matches the stored training data.*")
                    # Sample from predictive distribution
                    sampler = IIDNormalSampler(sample_shape=torch.Size([self.var_samples]))
                    # Get noise predictions (need to unstandardize)
                    noise_pred_std = noise_model.posterior(self.train_x).mean
                    log_var = noise_pred_std * saved_log_var_std + saved_log_var_mean
                    noise_var = log_var.exp().squeeze(-1)
                    # Create predictive distribution
                    mean_pred = hetero_model(self.train_x).mean
                    var_pred = hetero_model(self.train_x).variance + noise_var
                    
                    # Sample and estimate variance
                    samples = torch.randn(self.var_samples, len(self.train_y)) * torch.sqrt(var_pred) + mean_pred
                    observed_var = 0.5 * ((samples - self.train_y.unsqueeze(0))**2).mean(dim=0)
            
            # Clamp variance to reasonable range for numerical stability
            observed_var = torch.clamp(observed_var, min=1e-6, max=10.0)
        
        print(f'DID NOT REACH CONVERGENCE AFTER {self.max_iter} ITERATIONS')
        self.model = hetero_model
        self.noise_model = noise_model
        # Store the final standardization params
        self.noise_log_var_mean = saved_log_var_mean if 'saved_log_var_mean' in locals() else torch.tensor(0.0)
        self.noise_log_var_std = saved_log_var_std if 'saved_log_var_std' in locals() else torch.tensor(1.0)
    
    def posterior(self, X, observation_noise=False, posterior_transform=None, **kwargs):
        """
        Compute posterior distribution at test points.
        
        This is the key advantage over FixedNoiseGP: we can predict input-dependent
        noise variance at test points using the noise GP model.
        
        Args:
            X: Test inputs (BoTorch convention: capital X)
            observation_noise: If True, include predicted observation noise in variance
            posterior_transform: Optional posterior transform (for BoTorch compatibility)
            
        Returns:
            GPyTorchPosterior with mean and variance predictions
        """
        if self.model is None or self.noise_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Normalize test inputs if we normalized training inputs
        if self.norm_and_std and self.X_bounds is not None:
            X_norm = normalize(X, self.X_bounds)
        else:
            X_norm = X
        
        self.model.eval()
        self.noise_model.eval()
        
        with torch.no_grad():
            # Get mean and covariance from mean model
            pred_dist = self.model(X_norm)
            mean = pred_dist.mean
            covar = pred_dist.lazy_covariance_matrix  # Keep as lazy tensor
            
            # Get predicted noise variance from noise model
            # This is the key: noise_model can predict at NEW test points!
            noise_posterior = self.noise_model.posterior(X_norm)
            # Unstandardize and exp to get variance
            log_var_std = noise_posterior.mean
            log_var = log_var_std * self.noise_log_var_std + self.noise_log_var_mean
            predicted_noise = log_var.exp().clamp(min=MIN_INFERRED_NOISE_LEVEL).squeeze(-1)
            
            if observation_noise:
                # Add predicted observation noise to diagonal
                from linear_operator.operators import AddedDiagLinearOperator
                covar = AddedDiagLinearOperator(covar, DiagLinearOperator(predicted_noise))
            
            # Unstandardize predictions
            if self.norm_and_std:
                mean = mean * self.y_std + self.y_mean
                covar = covar * (self.y_std ** 2)
            
            mvn = MultivariateNormal(mean, covar)
        
        # Return GPyTorchPosterior for BoTorch compatibility
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        posterior = GPyTorchPosterior(mvn)
        
        # Apply posterior transform if provided
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        
        return posterior
    
    def get_predicted_noise_variance(self, x):
        """
        Get the predicted noise variance at test points.
        
        This is useful for diagnostics and visualization.
        
        Args:
            x: Test inputs (in original scale)
            
        Returns:
            Tensor of predicted noise variances (in original scale)
        """
        if self.noise_model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        
        # Normalize test inputs if we normalized training inputs
        if self.norm_and_std and self.X_bounds is not None:
            x_norm = normalize(x, self.X_bounds)
        else:
            x_norm = x
        
        self.noise_model.eval()
        
        with torch.no_grad():
            # Get predicted STANDARDIZED log-variance from noise model
            noise_posterior = self.noise_model.posterior(x_norm)
            log_var_std = noise_posterior.mean
            
            # Unstandardize the log-variance prediction
            log_var = log_var_std * self.noise_log_var_std + self.noise_log_var_mean
            
            # Exp to get variance (in standardized y space)
            predicted_noise_std_space = log_var.exp().clamp(min=MIN_INFERRED_NOISE_LEVEL)
            
            # Transform back to original scale
            # Variance in original space = variance in std space * (y_std^2)
            if self.norm_and_std:
                predicted_noise = predicted_noise_std_space * (self.y_std ** 2)
            else:
                predicted_noise = predicted_noise_std_space
        
        return predicted_noise
    
    def __call__(self, x):
        """Forward pass for compatibility."""
        if self.model is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return self.model(x)
    
    def train(self):
        """Set to training mode."""
        if self.model is not None:
            self.model.train()
        if self.noise_model is not None:
            self.noise_model.train()
    
    def eval(self):
        """Set to evaluation mode."""
        if self.model is not None:
            self.model.eval()
        if self.noise_model is not None:
            self.noise_model.eval()
