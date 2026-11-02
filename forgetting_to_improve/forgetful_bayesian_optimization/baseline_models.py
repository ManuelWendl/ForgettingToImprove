import torch
import gpytorch
import warnings
from typing import Tuple, Any
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.priors import GammaPrior
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood

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
    kernel: Any = None,
    device: str = 'cpu'
) -> Tuple[Any, Any]:
    """
    Initialize a Relevance Pursuit GP model.
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Kernel (covar_module) for the GP
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
    
    # Create kernel if not provided
    if kernel is None:
        n_dims = train_x_double.shape[-1]
        if n_dims == 1:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ).double()
        else:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            ).double()
    else:
        kernel = kernel.double()
    
    # Create relevance pursuit model with custom kernel
    rp_model = RobustRelevancePursuitSingleTaskGP(
        train_X=train_x_double,
        train_Y=train_y_double,
        covar_module=kernel,
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
    kernel: Any = None,
    noise_level: float = 0.1,
    device: str = 'cpu',
    global_bounds: torch.Tensor = None
) -> Tuple[Any, Any]:
    """
    Initialize a Warped GP model with input transformation (Kumaraswamy input warping).
    
    Uses a fresh Matern-5/2 kernel; the noise is learned (no fixed train_Yvar).
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Unused (a fresh kernel is created)
        noise_level: Sets the upper bound of the noise constraint (10 * noise_level)
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
    
    # Fresh Matern-5/2 kernel (hyperparameters are refit in every BO iteration)
    n_dims = train_x_double.shape[-1]
    
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
    
    # Same noise constraint as the standard GP
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
        # Use the problem bounds
        bounds_warp = global_bounds.double().to(device)
    else:
        # Otherwise use the bounds of the training data
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


def initialize_trimmed_mll_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: Any = None,
    noise_level: float = 0.1,
    device: str = 'cpu',
    max_outlier_fraction: float = 0.2,
    max_iterations: int = 50
) -> Tuple[Any, Any]:
    """
    Initialize a Trimmed MLL GP model that robustly handles outliers.
    
    Based on "Trimmed Maximum Likelihood Estimation for Robust Learning in Gaussian Processes"
    (https://proceedings.mlr.press/v161/kuss21a.html)
    
    This method jointly optimizes:
    1. GP hyperparameters (via marginal likelihood)
    2. Selection of inlier samples (removing up to max_outlier_fraction outliers)
    
    The algorithm iteratively:
    - Fits GP on current inlier set
    - Computes marginal likelihood contribution of each sample
    - Removes samples with lowest contributions (potential outliers)
    - Repeats until convergence
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Unused (a fresh kernel is created)
        noise_level: Initial noise level
        device: Device for computation
        max_outlier_fraction: Maximum fraction of samples to remove (default 0.2)
        max_iterations: Maximum optimization iterations
        
    Returns:
        Tuple of (mll, model) trained on inlier samples only
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for trimmed_mll method")
    
    print("Initializing Trimmed MLL GP model...")
    
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
    max_nr_outliers = int(n_train * max_outlier_fraction)
    min_nr_inliers = n_train - max_nr_outliers
    
    print(f"Total samples: {n_train}, Max outliers: {max_nr_outliers}, Min inliers: {min_nr_inliers}")
    
    # Fresh Matern-5/2 kernel
    base_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5, 
        ard_num_dims=n_dims,
        lengthscale_prior=GammaPrior(3.0, 1.5),
        lengthscale_constraint=GreaterThan(1e-4)
    ).double()
    
    kernel_trimmed = gpytorch.kernels.ScaleKernel(
        base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15)
    ).double()
    
    # Create likelihood with flexible noise constraints
    # Use GreaterThan instead of Interval to avoid constraint violation errors
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(2.0, 0.05),
        noise_constraint=GreaterThan(1e-6)
    ).double()
    
    # Initialize with all samples as inliers
    inlier_indices = torch.arange(n_train, dtype=torch.long, device=device)
    previous_inlier_indices = None
    previous_loss = torch.tensor(float('inf'))
    
    # Create initial model
    model = SingleTaskGP(
        train_X=train_x_double[inlier_indices],
        train_Y=train_y_double[inlier_indices],
        likelihood=likelihood,
        covar_module=kernel_trimmed
    )
    
    mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
    
    # Joint optimization loop (Algorithm 1 from paper)
    for iteration in range(max_iterations):
        # Set to training mode
        model.train()
        likelihood.train()
        
        # Create optimizer for current model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        
        # Optimize hyperparameters on current inlier set
        optimizer.zero_grad()
        output = model(train_x_double[inlier_indices])
        loss = -mll(output, train_y_double[inlier_indices].squeeze(-1))
        loss.backward()
        optimizer.step()
        
        current_loss = loss.item()
        
        print(f"Iteration {iteration}: Loss={current_loss:.4f}, Inliers={len(inlier_indices)}")
        
        # Compute marginal likelihood contribution for each sample
        # Using leave-one-out approximation: contribution ≈ -log p(y_i | y_{-i}, X)
        model.eval()
        likelihood.eval()
        
        with torch.no_grad():
            # Get predictive distribution at all training points
            pred = model(train_x_double)
            pred_mean = pred.mean
            pred_var = pred.variance + likelihood.noise
            
            # Compute negative log probability (lower is better for inliers)
            residuals = (train_y_double.squeeze(-1) - pred_mean) ** 2
            neg_log_prob = 0.5 * (torch.log(2 * torch.pi * pred_var) + residuals / pred_var)
        
        # Select new inlier set: keep samples with lowest neg_log_prob
        # (i.e., highest marginal likelihood contribution)
        sorted_indices = torch.argsort(neg_log_prob)
        new_inlier_indices = sorted_indices[:min_nr_inliers]
        new_inlier_indices = torch.sort(new_inlier_indices)[0]  # Sort for consistency
        
        # Check convergence: did inlier set change?
        if previous_inlier_indices is not None:
            if torch.equal(new_inlier_indices, previous_inlier_indices):
                print(f"Converged: Inlier set unchanged at iteration {iteration}")
                break
            
            # Check if new set improves marginal likelihood
            # Evaluate loss on new inlier set
            test_model = SingleTaskGP(
                train_X=train_x_double[new_inlier_indices],
                train_Y=train_y_double[new_inlier_indices],
                likelihood=likelihood,
                covar_module=kernel_trimmed
            )
            test_mll = ExactMarginalLogLikelihood(likelihood=test_model.likelihood, model=test_model)
            test_model.train()
            
            with torch.no_grad():
                test_output = test_model(train_x_double[new_inlier_indices])
                new_loss = -test_mll(test_output, train_y_double[new_inlier_indices].squeeze(-1)).item()
            
            if new_loss >= current_loss:
                # No improvement, use previous inlier set
                print(f"Converged: No improvement in marginal likelihood at iteration {iteration}")
                break
        
        # Update for next iteration
        previous_inlier_indices = inlier_indices.clone()
        inlier_indices = new_inlier_indices
        
        # Update model with new inlier set
        model.set_train_data(
            inputs=train_x_double[inlier_indices],
            targets=train_y_double[inlier_indices].squeeze(-1),
            strict=False
        )
    
    # Final model fit on selected inliers
    try:
        final_model = SingleTaskGP(
            train_X=train_x_double[inlier_indices],
            train_Y=train_y_double[inlier_indices],
            likelihood=likelihood,
            covar_module=kernel_trimmed
        )
        final_mll = ExactMarginalLogLikelihood(likelihood=final_model.likelihood, model=final_model)
        
        # Final hyperparameter optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(final_mll)
            except RuntimeError as e:
                if "constraint" in str(e).lower() or "prior" in str(e).lower():
                    print(f"  Warning: Hyperparameter fitting failed ({e}), using current values")
                else:
                    raise
        
        outlier_indices = torch.tensor([i for i in range(n_train) if i not in inlier_indices.tolist()])
        print(f"Trimmed MLL model fitted: {len(inlier_indices)} inliers, {len(outlier_indices)} outliers removed")
        
        return final_mll, final_model
    except Exception as e:
        print(f"  Error in final trimmed MLL model: {e}")
        print(f"  Falling back to standard GP")
        # Fallback: use all data with standard GP
        fallback_model = SingleTaskGP(
            train_X=train_x_double,
            train_Y=train_y_double,
            covar_module=kernel_trimmed,
            likelihood=likelihood
        )
        fallback_mll = ExactMarginalLogLikelihood(likelihood=fallback_model.likelihood, model=fallback_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(fallback_mll)
            except:
                pass  # Use default hyperparameters
        return fallback_mll, fallback_model


def initialize_student_t_gp_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    kernel: Any = None,
    noise_level: float = 0.1,
    device: str = 'cpu',
    degrees_of_freedom: float = 3.0
) -> Tuple[Any, Any]:
    """
    Initialize a robust GP model using iterative reweighting (simulating Student-t behavior).
    
    Uses Iteratively Reweighted Least Squares (IRLS) to down-weight outliers,
    which approximates the behavior of a Student-t likelihood. This provides
    robustness to outliers without requiring approximate inference methods.
    
    The algorithm:
    1. Fit a standard GP
    2. Compute residuals and down-weight points with large residuals
    3. Refit with weighted likelihood
    4. Iterate until convergence
    
    The degrees of freedom parameter controls tail heaviness:
    - Lower values (e.g., 3-5): More aggressive down-weighting of outliers
    - Higher values (e.g., >30): Approaches standard GP
    
    Args:
        train_x: Training inputs (n, d)
        train_y: Training targets (n, 1) or (n,)
        kernel: Unused (a fresh kernel is created)
        noise_level: Initial noise level
        device: Device for computation
        degrees_of_freedom: Degrees of freedom for robust weighting (default 3.0)
        
    Returns:
        Tuple of (mll, model) with robustly fitted hyperparameters
    """
    if not BOTORCH_AVAILABLE:
        raise ImportError("BoTorch is required for student_t method")
    
    print(f"Initializing Robust GP model (Student-t approximation with nu={degrees_of_freedom})...")
    
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
    
    # Fresh Matern-5/2 kernel
    base_kernel = gpytorch.kernels.MaternKernel(
        nu=2.5, 
        ard_num_dims=n_dims,
        lengthscale_prior=GammaPrior(3.0, 1.5),
        lengthscale_constraint=GreaterThan(1e-4)
    ).double()
    
    kernel_robust = gpytorch.kernels.ScaleKernel(
        base_kernel,
        outputscale_prior=GammaPrior(2.0, 0.15)
    ).double()
    
    # Create Gaussian likelihood with flexible constraint
    # Use GreaterThan instead of Interval to avoid constraint violation errors
    likelihood = GaussianLikelihood(
        noise_prior=GammaPrior(2.0, 0.05),
        noise_constraint=GreaterThan(1e-6)
    ).double()
    
    # Iteratively Reweighted Least Squares (IRLS) for robustness
    # This approximates Student-t likelihood behavior
    max_iterations = 5
    weights = torch.ones(n_train, dtype=torch.float64, device=device)
    
    # Initialize noise estimate (use initial likelihood noise or a default)
    try:
        current_noise = likelihood.noise.detach().clone().item()
    except:
        current_noise = 0.1  # Default noise level
    
    for iteration in range(max_iterations):
        try:
            # Create model with current weights by scaling the noise
            # Points with smaller weights get higher noise (less trust)
            # Clamp to prevent extreme values
            train_yvar = (current_noise / weights.clamp(min=0.01)).unsqueeze(-1)
            train_yvar = torch.clamp(train_yvar, min=1e-6, max=100.0)  # Prevent extreme variance
            
            # Check for NaN/Inf
            if torch.any(torch.isnan(train_yvar)) or torch.any(torch.isinf(train_yvar)):
                print(f"  Warning: Invalid train_Yvar detected, using uniform weights")
                train_yvar = torch.full_like(train_yvar, current_noise)
            
            model = SingleTaskGP(
                train_X=train_x_double,
                train_Y=train_y_double,
                train_Yvar=train_yvar,
                covar_module=kernel_robust,
                likelihood=likelihood
            )
            
            mll = ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit_gpytorch_mll(mll)
                except RuntimeError as e:
                    if "constraint" in str(e).lower() or "prior" in str(e).lower():
                        print(f"  Warning: Fitting failed in iteration {iteration + 1}, using current parameters")
                        break  # Exit IRLS loop and use current model
                    else:
                        raise
            
            # Update noise estimate for next iteration
            try:
                current_noise = model.likelihood.noise.detach().clone().item()
            except:
                pass  # Keep previous noise estimate
        except Exception as e:
            print(f"  Warning: IRLS iteration {iteration + 1} failed: {e}")
            print(f"  Continuing with previous weights...")
            break
        
        # Compute residuals
        model.eval()
        with torch.no_grad():
            try:
                pred = model(train_x_double)
                residuals = (train_y_double.squeeze(-1) - pred.mean)**2
                
                # Update weights using Student-t weighting formula:
                # w_i = (nu + d) / (nu + r_i^2 / sigma^2)
                # where r_i is the standardized residual
                sigma_sq = residuals.median()  # Robust scale estimate
                if sigma_sq < 1e-10:  # Avoid division by very small numbers
                    sigma_sq = torch.tensor(1e-6, dtype=torch.float64, device=device)
                
                standardized_resid_sq = residuals / (sigma_sq + 1e-6)
                
                # Student-t weights (down-weight outliers)
                weights = (degrees_of_freedom + 1) / (degrees_of_freedom + standardized_resid_sq)
                weights = torch.clamp(weights, min=0.01, max=1.0)  # Prevent extreme weights
                
                # Check for NaN/Inf
                if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
                    print(f"  Warning: Invalid weights detected, keeping previous weights")
                    weights = torch.ones(n_train, dtype=torch.float64, device=device)
            except Exception as e:
                print(f"  Warning: Weight update failed: {e}, keeping previous weights")
                # Keep previous weights
        
        model.train()
        
        if iteration < max_iterations - 1:
            print(f"  IRLS iteration {iteration + 1}: min_weight={weights.min():.3f}, max_weight={weights.max():.3f}")
    
    # Final model with learned weights
    try:
        # Use current noise estimate
        train_yvar_final = (current_noise / weights.clamp(min=0.01)).unsqueeze(-1)
        train_yvar_final = torch.clamp(train_yvar_final, min=1e-6, max=100.0)
        
        # Check for NaN/Inf
        if torch.any(torch.isnan(train_yvar_final)) or torch.any(torch.isinf(train_yvar_final)):
            print(f"  Warning: Invalid final train_Yvar, using uniform weights")
            train_yvar_final = torch.full((n_train, 1), current_noise, dtype=torch.float64, device=device)
        
        final_model = SingleTaskGP(
            train_X=train_x_double,
            train_Y=train_y_double,
            train_Yvar=train_yvar_final,
            covar_module=kernel_robust,
            likelihood=likelihood
        )
        
        final_mll = ExactMarginalLogLikelihood(likelihood=final_model.likelihood, model=final_model)
        
        # Final hyperparameter optimization
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                fit_gpytorch_mll(final_mll)
            except RuntimeError as e:
                if "constraint" in str(e).lower() or "prior" in str(e).lower():
                    print(f"  Warning: Final hyperparameter fitting failed, using current values")
                else:
                    raise
        
        print(f"Robust GP model fitted (IRLS converged, {(weights < 0.5).sum()}/{n_train} points down-weighted)")
        return final_mll, final_model
    except Exception as e:
        print(f"  Error in final model creation: {e}")
        print(f"  Falling back to standard GP without robustness")
        # Fallback: return a standard GP model
        fallback_model = SingleTaskGP(
            train_X=train_x_double,
            train_Y=train_y_double,
            covar_module=kernel_robust,
            likelihood=likelihood
        )
        fallback_mll = ExactMarginalLogLikelihood(likelihood=fallback_model.likelihood, model=fallback_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit_gpytorch_mll(fallback_mll)
        return fallback_mll, fallback_model
