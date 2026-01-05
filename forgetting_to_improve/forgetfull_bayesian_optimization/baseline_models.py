import torch
import gpytorch
import warnings
from typing import Tuple, Any, Dict
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

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
    import gpytorch
    from gpytorch.priors import GammaPrior
    from gpytorch.constraints import GreaterThan, Interval
    from gpytorch.likelihoods import GaussianLikelihood
    
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
    Initialize a Heteroscedastic GP model.
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        mean_kernel: Kernel for the mean GP
        noise_kernel: Kernel for the noise GP (optional)
        device: Device for computation
        
    Returns:
        Tuple of (None, model) - mll is None as it's managed internally
    """
    print("Initializing Heteroscedastic GP model...")
    
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    
    # Ensure train_y is 1D
    if train_y.ndim > 1:
        train_y = train_y.squeeze(-1)
    
    # Create noise kernel if not provided
    if noise_kernel is None:
        n_dims = train_x.shape[-1]
        if n_dims == 1:
            noise_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ).to(device)
        else:
            noise_kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            ).to(device)
    
    # Create heteroscedastic model
    hetero_model = HeteroscedasticGPModel(
        train_x=train_x,
        train_y=train_y,
        kernel=mean_kernel,
        noise_kernel=noise_kernel,
        device=device
    )
    
    # Fit the model
    hetero_model.fit(max_iter=50, lr=0.1)
    
    print("Heteroscedastic GP model fitted successfully")
    return None, hetero_model