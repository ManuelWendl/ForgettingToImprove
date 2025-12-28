import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    """Simple Exact GP model for regression."""
    
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class HeteroscedasticGPModel:
    """
    Heteroscedastic GP model with separate GPs for mean and noise variance.
    
    This implementation uses two GPs:
    1. A primary GP for modeling the mean function
    2. A secondary GP for modeling the log noise variance as a function of inputs
    
    This allows the model to learn input-dependent (heteroscedastic) noise.
    """
    
    def __init__(self, train_x, train_y, kernel, noise_kernel=None, device='cpu'):
        """
        Args:
            train_x: Training inputs
            train_y: Training targets
            kernel: Kernel for the mean GP
            noise_kernel: Kernel for the noise GP (if None, uses a simpler kernel)
            device: Device to use for computations
        """
        self.train_x = train_x.to(device)
        self.train_y = train_y.to(device)
        self.device = device
        
        # Initialize with homoscedastic noise estimate
        initial_noise = torch.var(train_y) * 0.1
        
        # Mean GP with fixed noise (will be updated iteratively)
        self.mean_likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
            noise=torch.full_like(train_y, initial_noise),
            learn_additional_noise=False
        )
        self.mean_gp = ExactGPModel(train_x, train_y, self.mean_likelihood, kernel)
        
        # Noise GP for learning log noise variance
        if noise_kernel is None:
            # Use a simpler kernel for noise (typically smoother)
            n_dims = train_x.shape[-1] if train_x.ndim > 1 else 1
            if n_dims == 1:
                noise_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel()
                )
            else:
                noise_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
                )
        
        # Estimate initial log noise from residuals (after fitting mean GP once)
        self.noise_likelihood = gpytorch.likelihoods.GaussianLikelihood()
        
        # Initialize noise GP targets as log of initial noise estimate
        log_noise_targets = torch.full_like(train_y, torch.log(initial_noise))
        self.noise_gp = ExactGPModel(train_x, log_noise_targets, self.noise_likelihood, noise_kernel)
        
        self.mean_gp = self.mean_gp.to(device)
        self.noise_gp = self.noise_gp.to(device)
    
    def fit(self, max_iter=50, lr=0.1):
        """
        Fit the heteroscedastic GP using alternating optimization.
        
        Args:
            max_iter: Maximum number of iterations for alternating optimization
            lr: Learning rate for optimization
        """
        for iteration in range(max_iter):
            # Step 1: Fit mean GP with current noise estimates
            self.mean_gp.train()
            self.mean_likelihood.train()
            
            optimizer_mean = torch.optim.Adam(self.mean_gp.parameters(), lr=lr)
            mll_mean = gpytorch.mlls.ExactMarginalLogLikelihood(self.mean_likelihood, self.mean_gp)
            
            # A few steps for mean GP
            for _ in range(10):
                optimizer_mean.zero_grad()
                output = self.mean_gp(self.train_x)
                loss = -mll_mean(output, self.train_y)
                loss.backward()
                optimizer_mean.step()
            
            # Step 2: Compute residuals and fit noise GP
            self.mean_gp.eval()
            with torch.no_grad():
                pred = self.mean_gp(self.train_x)
                residuals = (self.train_y - pred.mean).pow(2)
                # Add small constant for numerical stability
                log_noise_targets = torch.log(residuals + 1e-6)
            
            # Update noise GP targets
            self.noise_gp.set_train_data(self.train_x, log_noise_targets, strict=False)
            
            self.noise_gp.train()
            self.noise_likelihood.train()
            
            optimizer_noise = torch.optim.Adam(self.noise_gp.parameters(), lr=lr)
            mll_noise = gpytorch.mlls.ExactMarginalLogLikelihood(self.noise_likelihood, self.noise_gp)
            
            # A few steps for noise GP
            for _ in range(10):
                optimizer_noise.zero_grad()
                output = self.noise_gp(self.train_x)
                loss = -mll_noise(output, log_noise_targets)
                loss.backward()
                optimizer_noise.step()
            
            # Step 3: Update the fixed noise in mean GP likelihood
            self.noise_gp.eval()
            with torch.no_grad():
                noise_pred = self.noise_gp(self.train_x)
                predicted_noise = torch.exp(noise_pred.mean)
                # Clip noise to reasonable range for numerical stability
                predicted_noise = torch.clamp(predicted_noise, min=1e-6, max=1e2)
                self.mean_likelihood.noise = predicted_noise
        
        # Final evaluation mode
        self.mean_gp.eval()
        self.noise_gp.eval()
        self.mean_likelihood.eval()
        self.noise_likelihood.eval()
    
    def predict(self, x_test):
        """
        Make predictions at test points with heteroscedastic noise estimates.
        
        Args:
            x_test: Test inputs
            
        Returns:
            Tuple of (mean, variance, aleatoric_noise) where aleatoric_noise
            is the predicted input-dependent noise variance
        """
        x_test = x_test.to(self.device)
        
        self.mean_gp.eval()
        self.noise_gp.eval()
        
        with torch.no_grad():
            # Predict mean and epistemic variance
            mean_pred = self.mean_gp(x_test)
            mean = mean_pred.mean
            epistemic_var = mean_pred.variance
            
            # Predict aleatoric (noise) variance
            noise_pred = self.noise_gp(x_test)
            aleatoric_var = torch.exp(noise_pred.mean)
            
        return mean, epistemic_var, aleatoric_var
    
    def __call__(self, x):
        """Forward pass returns the mean GP distribution."""
        return self.mean_gp(x)
    
    def train(self):
        """Set to training mode."""
        self.mean_gp.train()
        self.noise_gp.train()
        self.mean_likelihood.train()
        self.noise_likelihood.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.mean_gp.eval()
        self.noise_gp.eval()
        self.mean_likelihood.eval()
        self.noise_likelihood.eval()