import torch
import gpytorch
import warnings
from typing import Tuple, Any, Callable, Optional
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from torch import Tensor
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood

# Import baseline models from forgetting_to_improve
from ..forgetting_to_improve.models import HeteroscedasticGPModel

# Import BoTorch models for baselines
try:
    from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.input import Warp
    from gpytorch.priors.torch_priors import LogNormalPrior
    BOTORCH_AVAILABLE = True
except ImportError:
    BOTORCH_AVAILABLE = False
    print("Warning: BoTorch not available. Baseline methods will not work.")


def initialize_relevance_pursuit_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: Any = None,  # Keep for API compatibility but ignore
    device: str = 'cpu'
) -> Tuple[Any, Any]:
    """
    Initialize a Relevance Pursuit GP model.
    
    Note: kernel parameter is ignored. RobustRelevancePursuitSingleTaskGP creates 
    its own kernel internally for fair comparison with other papers.
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Ignored (kept for API compatibility)
        device: Device for computation
        
    Returns:
        Tuple of (mll, model)
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for relevance_pursuit method")
    
    print("Initializing Relevance Pursuit model...")
    
    # Convert to double precision for BoTorch
    train_x_double = train_x.double().to(device)
    train_y_double = train_y.double().to(device)
    
    # Handle NaN/Inf
    train_x_double = torch.nan_to_num(train_x_double, nan=0.0, posinf=0.0, neginf=0.0)
    train_y_double = torch.nan_to_num(train_y_double, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure train_y has shape (n, 1)
    if train_y_double.ndim == 1:
        train_y_double = train_y_double.unsqueeze(-1)
    
    # Prior for number of outliers
    n_train = len(train_x_double)
    prior_mean_of_support = int(0.2 * n_train)
    
    # Create relevance pursuit model WITHOUT custom kernel - let BoTorch create its own
    # This is how relevance pursuit is typically used in other papers
    rp_model = RobustRelevancePursuitSingleTaskGP(
        train_X=train_x_double,
        train_Y=train_y_double,
        convex_parameterization=True,
        cache_model_trace=False,
        prior_mean_of_support=prior_mean_of_support
    )
    
    # Fit using relevance pursuit
    mll_rp = ExactMarginalLogLikelihood(likelihood=rp_model.likelihood, model=rp_model)
    
    numbers_of_outliers = [0, int(0.05*n_train), int(0.1*n_train), int(0.15*n_train), 
                          int(0.2*n_train), int(0.3*n_train), int(0.4*n_train), 
                          int(0.5*n_train), int(0.75*n_train), n_train]
    rp_kwargs = {
        "numbers_of_outliers": numbers_of_outliers,
        "optimizer_kwargs": {"options": {"maxiter": 1024}},
    }
    
    # Suppress warnings during fitting
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='.*Robust rho not applied.*')
        fit_gpytorch_mll(mll_rp, **rp_kwargs)
    
    print("Relevance Pursuit model fitted successfully")
    return mll_rp, rp_model


def initialize_warped_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: Any = None,  # Now used - Matern kernel from config
    noise_level: float = 0.1,
    device: str = 'cpu',
    global_bounds: torch.Tensor = None
) -> Tuple[Any, Any]:
    """
    Initialize a Warped GP model with input transformation.
    
    Uses the Matern kernel from config (same as standard GP) and lets the model
    learn noise internally without fixed train_Yvar.
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Matern kernel from config (same as standard GP uses)
        noise_level: Ignored - model learns noise internally
        device: Device for computation
        global_bounds: Global problem bounds (2, d) for input warping
        
    Returns:
        Tuple of (mll, model)
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for warped_gp method")
    
    print("Initializing Warped GP model...")
    
    # Convert to double precision
    train_x_double = train_x.double().to(device)
    train_y_double = train_y.double().to(device)
    
    # Handle NaN/Inf
    train_x_double = torch.nan_to_num(train_x_double, nan=0.0, posinf=0.0, neginf=0.0)
    train_y_double = torch.nan_to_num(train_y_double, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure train_y has shape (n, 1)
    if train_y_double.ndim == 1:
        train_y_double = train_y_double.unsqueeze(-1)
    
    # Create fresh Matern kernel with same priors as standard GP (not pre-optimized)
    
    n_dims = train_x_double.shape[-1]
    
    # Create Matern kernel with EXACT same priors as used in setup_experiment.py
    base_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5, 
        ard_num_dims=n_dims,
        lengthscale_prior=GammaPrior(3.0, 1.5),
        lengthscale_constraint=GreaterThan(1e-4)
    ).double()
    
    kernel_warped = gpytorch.kernels.ScaleKernel(
        base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15)
    ).double()
    
    # Create likelihood with EXACT same noise constraints as standard GP
    if noise_level > 0:
        likelihood_warped = GaussianLikelihood(
            noise_prior=GammaPrior(1.5, 1.0),
            noise_constraint=Interval(1e-6, noise_level * 10.0)
        ).double()
    else:
        likelihood_warped = GaussianLikelihood(
            noise_prior=GammaPrior(1.5, 1.0),
            noise_constraint=Interval(1e-6, 1e-2)
        ).double()
    
    # Determine bounds for input warping
    n_dims = train_x_double.shape[-1]
    if global_bounds is not None:
        # Use global problem bounds (proper approach)
        bounds_warp = global_bounds.double().to(device)
    else:
        # Fallback to training data bounds (for backward compatibility)
        if n_dims == 1:
            x_min = train_x_double.min().unsqueeze(0)
            x_max = train_x_double.max().unsqueeze(0)
        else:
            x_min = train_x_double.min(dim=0)[0]
            x_max = train_x_double.max(dim=0)[0]
        bounds_warp = torch.stack([x_min, x_max])
    
    # Initialize Warp transformation
    warp_tf = Warp(
        d=n_dims,
        indices=list(range(n_dims)),
        concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
        concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
        bounds=bounds_warp
    )
    
    # Create warped GP model with Matern kernel and SAME likelihood constraints as standard GP
    warped_model = SingleTaskGP(
        train_X=train_x_double,
        train_Y=train_y_double,
        likelihood=likelihood_warped,
        covar_module=kernel_warped,
        input_transform=warp_tf
    )
    
    # Fit warped GP
    mll_warped = ExactMarginalLogLikelihood(likelihood=warped_model.likelihood, model=warped_model)
    fit_gpytorch_mll(mll_warped)
    
    print("Warped GP model fitted successfully")
    return mll_warped, warped_model


def initialize_heteroscedastic_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    mean_kernel: Any,
    noise_kernel: Any = None,
    device: str = 'cpu'
) -> Tuple[None, HeteroscedasticGPModel]:
    """
    Initialize a Most Likely Heteroscedastic GP model.
    
    This implements the algorithm from:
    "Most Likely Heteroscedastic Gaussian Process Regression"
    http://people.csail.mit.edu/kersting/papers/kersting07icml_mlHetGP.pdf
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        mean_kernel: Kernel for the mean GP
        noise_kernel: Not used (kept for compatibility)
        device: Device for computation
        
    Returns:
        Tuple of (None, model) - mll is None as it's managed internally
    """
    print("Initializing Most Likely Heteroscedastic GP model...")
    
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    # Ensure train_y is 1D
    if train_y.ndim > 1:
        train_y = train_y.squeeze(-1)
    
    # Add priors to mean kernel if not already present
    if not hasattr(mean_kernel.base_kernel, 'lengthscale_prior'):
        mean_kernel.base_kernel.lengthscale_prior = GammaPrior(3.0, 1.5)
        mean_kernel.base_kernel.lengthscale_constraint = GreaterThan(1e-4)
    if not hasattr(mean_kernel, 'outputscale_prior'):
        mean_kernel.outputscale_prior = GammaPrior(2.0, 0.15)
    
    # Create heteroscedastic model using the Most Likely approach
    hetero_model = HeteroscedasticGPModel(
        train_x=train_x,
        train_y=train_y,
        kernel=mean_kernel,
        max_iter=25,  # Match the notebook default
        tol=1e-04,
        var_estimate='paper',  # Use the paper's sampling-based method
        var_samples=1000,
        norm_and_std=True  # Always normalize and standardize for stability
    )
    
    # Fit the model (this implements the full Most Likely algorithm)
    hetero_model.fit()
    
    # Set to training mode for acquisition function optimization
    hetero_model.train()
    
    print("Most Likely Heteroscedastic GP model fitted successfully")
    return None, hetero_model


def initialize_modulating_surrogates_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: Any = None,
    noise_level: float = 0.1,
    device: str = 'cpu',
    num_mcmc_samples: int = 100,
    num_warmup: int = 20
) -> Tuple[Any, Any]:
    """
    Initialize a Modulating Surrogates GP model using Latent GP (LGP).
    
    Based on "Modulating Surrogates for Bayesian Optimization" paper.
    
    This implements the proper LGP approach where:
    1. Each training point has latent variables h_n ~ N(0, σ²_h I)
    2. Posterior inference over H and θ is performed via MCMC (HMC/NUTS)
    3. Acquisition function uses Monte Carlo approximation over posterior samples
    
    The latent variables allow the model to learn which training points are
    unreliable/detrimental and down-weight them automatically.
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Matern kernel from config (same as standard GP uses)
        noise_level: Noise level for the likelihood
        device: Device for computation
        num_mcmc_samples: Number of MCMC posterior samples to draw
        num_warmup: Number of MCMC warmup/burn-in steps
        
    Returns:
        Tuple of (None, model) - mll is None as MCMC handles inference
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for modulating_surrogates method")
    
    from ..forgetting_to_improve.models import create_latent_gp_model, LatentGPModel
    
    print("Initializing Latent GP (Modulating Surrogates) model...")
    
    # Convert to double precision
    train_x_double = train_x.double().to(device)
    train_y_double = train_y.double().to(device)
    
    # Handle NaN/Inf
    train_x_double = torch.nan_to_num(train_x_double, nan=0.0, posinf=0.0, neginf=0.0)
    train_y_double = torch.nan_to_num(train_y_double, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure train_y has shape (n, 1)
    if train_y_double.ndim == 1:
        train_y_double = train_y_double.unsqueeze(-1)
    
    n_train, n_dims = train_x_double.shape
    
    # Create Matern kernel with ARD for augmented space (d + latent_dim dimensions)
    latent_dim = 1  # One latent variable per training point
    base_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5, 
        ard_num_dims=n_dims + latent_dim,  # +latent_dim for the latent variables
        lengthscale_prior=GammaPrior(3.0, 1.5),
        lengthscale_constraint=GreaterThan(1e-4)
    ).double()
    
    kernel_modulating = gpytorch.kernels.ScaleKernel(
        base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15)
    ).double()
    
    # Create likelihood with same noise constraints as standard GP
    if noise_level > 0:
        likelihood_modulating = GaussianLikelihood(
            noise_prior=GammaPrior(1.5, 1.0),
            noise_constraint=Interval(1e-6, noise_level * 10.0)
        ).double()
    else:
        likelihood_modulating = GaussianLikelihood(
            noise_prior=GammaPrior(1.5, 1.0),
            noise_constraint=Interval(1e-6, 1e-2)
        ).double()
    
    # Create Latent GP model with MCMC
    # Prior std for latent variables (σ_h in the paper)
    sigma_h = 1.0
    
    try:
        modulating_model = create_latent_gp_model(
            train_x=train_x_double,
            train_y=train_y_double,
            covar_module=kernel_modulating,
            likelihood=likelihood_modulating,
            latent_dim=latent_dim,
            sigma_h=sigma_h,
            num_mcmc_samples=num_mcmc_samples,
            num_warmup=num_warmup
        )
        print("Latent GP (Modulating Surrogates) model fitted successfully via MCMC")
    except Exception as e:
        print(f"Warning: MCMC failed ({e}). Falling back to simplified version...")
        # Fallback to simpler approach without full MCMC
        modulating_model = LatentGPModel(
            train_X=train_x_double,
            train_Y=train_y_double,
            covar_module=kernel_modulating,
            likelihood=likelihood_modulating,
            latent_dim=latent_dim,
            sigma_h=sigma_h,
            num_mcmc_samples=100,  # Single sample (MAP estimate)
            num_warmup=100
        )
        # Use zero latent variables as fallback
        modulating_model.mcmc_samples_h = torch.zeros(
            1, n_train, latent_dim,
            dtype=train_x_double.dtype,
            device=train_x_double.device
        )
    
    return None, modulating_model

class MCLatentGPAcquisitionFunction:
    """
    Wrapper for acquisition functions that uses Monte Carlo approximation
    over latent variable posterior samples.
    
    Following the LGP paper:
    α(x*) ≈ (1/M) Σᵢ α̂(x*, Hᵢ, θᵢ)
    
    where {Hᵢ, θᵢ} are MCMC samples from the posterior.
    """
    
    def __init__(
        self,
        base_acq_func_factory: Callable,
        latent_gp_model,
        X_baseline: Tensor,
        num_samples: Optional[int] = None
    ):
        """
        Args:
            base_acq_func_factory: Factory function that creates acquisition function
                                   given a model (e.g., lambda model: UpperConfidenceBound(model, beta=0.1))
            latent_gp_model: LatentGPModel with MCMC samples
            X_baseline: Baseline points for acquisition (n, d) - original dimension
            num_samples: Number of MCMC samples to use (None = use all)
        """
        self.base_acq_func_factory = base_acq_func_factory
        self.latent_gp_model = latent_gp_model
        self.X_baseline = X_baseline
        
        # Get MCMC samples of both H and θ
        h_samples, theta_samples = latent_gp_model.get_posterior_samples()
        
        if num_samples is not None and num_samples < len(theta_samples):
            # Subsample if requested
            indices = torch.randperm(len(theta_samples))[:num_samples].tolist()
            self.h_samples = h_samples[indices]
            self.theta_samples = [theta_samples[i] for i in indices]
        else:
            self.h_samples = h_samples
            self.theta_samples = theta_samples
        
        self.num_samples = len(self.theta_samples)
    
    def __call__(self, X: Tensor) -> Tensor:
        """
        Evaluate acquisition function using Monte Carlo approximation.
        
        For each MCMC sample (Hᵢ, θᵢ):
        1. Create a fresh GP with training data augmented by Hᵢ and hyperparameters θᵢ
        2. Evaluate acquisition function α̂(x*, Hᵢ, θᵢ)
        3. Average over all samples: α(x*) ≈ (1/M) Σᵢ α̂(x*, Hᵢ, θᵢ)
        
        Args:
            X: Candidate points (batch_size, d) - original dimension
            
        Returns:
            Acquisition values (batch_size,)
        """
        from gpytorch.kernels import RBFKernel, ScaleKernel
        from gpytorch.likelihoods import GaussianLikelihood
        
        acq_values = torch.zeros(X.shape[0], dtype=X.dtype, device=X.device)
        
        # Monte Carlo approximation: average over posterior samples of (H, θ)
        for i in range(self.num_samples):
            # Get this sample's latent variables and hyperparameters
            h_sample = self.h_samples[i]  # (n, latent_dim)
            theta_sample = self.theta_samples[i]  # dict with 'lengthscale', 'outputscale', 'noise'
            
            # Create fresh GP model with augmented training data
            X_augmented = torch.cat([self.latent_gp_model.train_X_original, h_sample], dim=-1)
            Y_train = self.latent_gp_model.train_Y_original
            if Y_train.ndim == 1:
                Y_train = Y_train.unsqueeze(-1)
            
            # Create kernel with sampled hyperparameters
            base_kernel = RBFKernel(ard_num_dims=X_augmented.shape[-1])
            base_kernel.lengthscale = theta_sample['lengthscale'].detach()
            
            covar_module = ScaleKernel(base_kernel)
            covar_module.outputscale = theta_sample['outputscale'].detach()
            
            # Create likelihood with sampled noise
            likelihood = GaussianLikelihood()
            likelihood.noise = theta_sample['noise'].detach()
            
            # Create a fresh SingleTaskGP instance with sampled hyperparameters
            fresh_model = SingleTaskGP(
                train_X=X_augmented.double(),
                train_Y=Y_train.double(),
                covar_module=covar_module,
                likelihood=likelihood
            )
            
            # Set to eval mode
            fresh_model.eval()
            fresh_model.likelihood.eval()
            
            # Create acquisition function for this model
            acq_func = self.base_acq_func_factory(model=fresh_model, X_baseline=self.X_baseline)
            
            # Evaluate acquisition - augment X with zeros for test points
            h_test = torch.zeros(*X.shape[:-1], self.latent_gp_model.latent_dim, dtype=X.dtype, device=X.device)
            X_augmented_test = torch.cat([X, h_test], dim=-1)
            
            # Evaluate with augmented test points
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                acq_val = acq_func(X_augmented_test.unsqueeze(-2)).squeeze(-1)
            
            acq_values += acq_val
        
        # Average over samples
        acq_values /= self.num_samples
        
        return acq_values


def create_mc_latent_acquisition(
    acq_func_factory: Callable,
    latent_gp_model,
    X_baseline: Tensor,
    num_samples: Optional[int] = None
):
    """
    Create a Monte Carlo acquisition function for a Latent GP model.
    
    Args:
        acq_func_factory: Function that creates acquisition function given a model
        latent_gp_model: Fitted LatentGPModel with MCMC samples
        X_baseline: Training inputs in original dimension (n, d)
        num_samples: Number of MC samples to use
        
    Returns:
        Callable acquisition function
    """
    return MCLatentGPAcquisitionFunction(
        base_acq_func_factory=acq_func_factory,
        latent_gp_model=latent_gp_model,
        X_baseline=X_baseline,
        num_samples=num_samples
    )
