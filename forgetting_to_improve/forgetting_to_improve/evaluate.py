import torch
import gpytorch
import numpy as np
import warnings
import time
from typing import Tuple, Optional, List, Union, Callable
from .helper.plot_gp import plot_predictions, plot_curvature_bounds
from .helper.error import calculate_prediction_errors
from .helper.kernel_factory import create_kernel_from_config
from .helper.data_loader import load_dataset
from .helper.aux import filter_samples, rand_sample
from .models import ExactGPModel, HeteroscedasticGPModel
from .optimize import (
    sequentially_optimize_samples,
    batch_optimize_samples,
)

# BoTorch imports for relevance pursuit, warped GP, and heteroscedastic GP
BOTORCH_AVAILABLE = False
RobustRelevancePursuitSingleTaskGP = None
SingleTaskGP = None
Warp = None
fit_gpytorch_mll = None
ExactMarginalLogLikelihood = None
LogNormalPrior = None

try:
    from botorch.models.robust_relevance_pursuit_model import RobustRelevancePursuitSingleTaskGP # type: ignore
    from botorch.models import SingleTaskGP # type: ignore
    from botorch.models.transforms.input import Warp # type: ignore
    from botorch.fit import fit_gpytorch_mll # type: ignore
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors.torch_priors import LogNormalPrior
    BOTORCH_AVAILABLE = True
    print("BoTorch successfully imported for relevance pursuit, warped GP, and heteroscedastic GP")
except ImportError as e:
    print(f"Warning: Could not import BoTorch components: {e}")
    print("BoTorch methods (relevance_pursuit, warped_gp, heteroscedastic_gp) will not be available.")
except Exception as e:
    print(f"Warning: Unexpected error importing BoTorch: {e}")
    print("BoTorch methods (relevance_pursuit, warped_gp, heteroscedastic_gp) will not be available.")


def evaluate(
    method: str,
    objective_func: Union[Callable, str],
    noise: str,
    A_limits: Union[Tuple, List],
    X_limits: Union[Tuple, List],
    num_samples: int,
    num_train_samples: int,
    num_A_samples: int,
    seed: int = 0,
    fixed_kernel: bool = True,
    show_plots: bool = False,
    show_hessian: bool = False,
    calculate_convexity: bool = False,
    objective_type: str = 'function',
    feature_subset: Optional[Union[int, List[int]]] = None,
    kernel_config: Optional[List[dict]] = None,
    gp_alpha: float = 0.01,
    device: str = 'cpu',
    return_plot_data: bool = False
) -> Union[Tuple[dict, int], Tuple[dict, int, dict]]:
    """
    Evaluate the GP optimization method using GPyTorch.
    
    Args:
        method: Optimization method ('none', 'sequential', 'batch', 'targetSampling')
        objective_func: Function to optimize or dataset name
        noise: Noise type ('joint', 'epistemic', 'aleatoric')
        A_limits: Limits for active learning region
        X_limits: Limits for full domain
        num_samples: Number of samples to generate
        num_train_samples: Number of initial training samples
        num_A_samples: Number of active learning samples
        seed: Random seed
        fixed_kernel: Whether to fix kernel hyperparameters
        show_plots: Whether to show plots
        show_hessian: Whether to show Hessian
        calculate_convexity: Whether to compute curvature bounds m_epi and M_ale
        objective_type: Type of objective ('function' or 'dataset')
        feature_subset: Subset of features to use
        kernel_config: Kernel configuration
        gp_alpha: GP noise parameter
        device: Device for computation ('cpu' or 'cuda')
        return_plot_data: Whether to return plotting data (for comparison plots)
        
    Returns:
        errors: Dictionary of error metrics
        num_optimized_samples: Number of samples after optimization
        plot_data: (Optional) Dictionary with plotting data if return_plot_data=True
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    start_time = time.time()
    
    device = torch.device(device)
    
    if objective_type == 'function':
        # For functions, assume 1D for now (can be extended)
        if not isinstance(X_limits[0], (list, tuple)):
            # 1D case
            x = np.linspace(X_limits[0], X_limits[1], num_samples)
            x = x.reshape(-1, 1)  # Ensure 2D shape
        else:
            # Multidimensional case - generate grid or random samples
            n_dims = len(X_limits)
            x = np.random.uniform(
                low=[lim[0] for lim in X_limits],
                high=[lim[1] for lim in X_limits],
                size=(num_samples, n_dims)
            )
        
        # Apply objective function
        if x.shape[1] == 1:
            y_test = objective_func(x.flatten())
        else:
            y_test = np.array([objective_func(xi) for xi in x])
        
        y = y_test + np.random.normal(0, 0.01, y_test.shape)
        
        # Use the original objective function for plotting
        plot_func = objective_func
    
    elif objective_type == 'dataset':
        # Load real dataset
        x_data, y_data = load_dataset(objective_func)
        
        # Handle feature selection
        if feature_subset is not None:
            if isinstance(feature_subset, int):
                feature_subset = [feature_subset]
            x_data = x_data[:, feature_subset]
        
        # Ensure x_data is 2D
        if x_data.ndim == 1:
            x_data = x_data.reshape(-1, 1)
        
        x = x_data
        y = y_data
        n_dims = x.shape[1]
        
        # Update limits based on actual data if not specified or using defaults
        if X_limits == (-10, 10) or (isinstance(X_limits, list) and len(X_limits) == 2 and not isinstance(X_limits[0], (list, tuple))):
            # Default limits - auto-adjust
            if n_dims == 1:
                X_limits = (float(x.min()), float(x.max()))
            else:
                X_limits = [(float(x[:, i].min()), float(x[:, i].max())) for i in range(n_dims)]
        
        if A_limits == (-3, 3) or (isinstance(A_limits, list) and len(A_limits) == 2 and not isinstance(A_limits[0], (list, tuple))):
            # Default limits - auto-adjust to central region
            if n_dims == 1:
                x_range = x.max() - x.min()
                center = (x.max() + x.min()) / 2
                A_limits = (float(center - x_range * 0.3), float(center + x_range * 0.3))
            else:
                A_limits = []
                for i in range(n_dims):
                    x_range = x[:, i].max() - x[:, i].min()
                    center = (x[:, i].max() + x[:, i].min()) / 2
                    A_limits.append((float(center - x_range * 0.3), float(center + x_range * 0.3)))
        
        # Create a plotting function that interpolates the data (for 1D case)
        if n_dims == 1:
            def plot_func(x_plot):
                return np.interp(x_plot, x.flatten(), y)
        else:
            plot_func = None  # No plotting for multidimensional case
        
        # Trim data to match num_samples if dataset is larger
        if len(x) > num_samples:
            indices = np.random.choice(len(x), size=num_samples, replace=False)
            x = x[indices]
            y = y[indices]
    
    else:
        raise ValueError(f"Invalid objective_type: {objective_type}. Must be 'function' or 'dataset'")
    
    # Generate active learning region samples
    n_dims = x.shape[1]
    if n_dims == 1:
        x_A = np.linspace(A_limits[0], A_limits[1], num_A_samples).reshape(-1, 1)
    else:
        # For multidimensional case, generate samples within A_limits hypercube
        if not isinstance(A_limits[0], (list, tuple)):
            # Same limits for all dimensions
            A_limits = [A_limits] * n_dims
        
        x_A = np.random.uniform(
            low=[lim[0] for lim in A_limits],
            high=[lim[1] for lim in A_limits],
            size=(num_A_samples, n_dims)
        )
    
    # Initial training samples
    x_samples, y_samples = rand_sample(x, y, n_samples=num_train_samples)
    
    # Sort samples for better visualization (only for 1D)
    if n_dims == 1:
        sorted_indices = np.argsort(x_samples.flatten())
        x_samples = x_samples[sorted_indices]
        y_samples = y_samples[sorted_indices]
    
    if kernel_config is None:
        # Default kernel configuration (backward compatibility)
        if n_dims == 1:
            kernel_config = [
                {'type': 'rbf', 'length_scale': 1.0},
                {'type': 'white', 'noise_level': gp_alpha}
            ]
        else:
            kernel_config = [
                {'type': 'dot_product'},
                {'type': 'rbf', 'length_scale': 1.0},
                {'type': 'white', 'noise_level': gp_alpha}
            ]

    kernel, white_noise_level = create_kernel_from_config(kernel_config, n_dims, fixed_kernel, device)
    
    # Convert to torch tensors
    x_samples_torch = torch.from_numpy(x_samples).float().to(device)
    y_samples_torch = torch.from_numpy(y_samples).float().to(device)
    x_A_torch = torch.from_numpy(x_A).float().to(device)
    
    # Create likelihood and model
    # If white kernel is specified, use it; otherwise use gp_alpha
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if white_noise_level is not None:
        likelihood.noise = white_noise_level
    else:
        likelihood.noise = gp_alpha
    
    # Note: likelihood noise is always trainable (will be optimized)    
    model = ExactGPModel(x_samples_torch, y_samples_torch, likelihood, kernel)
    model = model.to(device)
    likelihood = likelihood.to(device)
    
    # Set to eval mode for predictions
    model.eval()
    likelihood.eval()
    
    # Optimize the training samples by removing detrimental ones
    curvature_history = {'m_epi': [], 'M_ale': [], 'ratio': [], 'n_samples': []}  # Initialize for all methods
    
    if method == 'none':
        x_samples_opt = x_samples
        y_samples_opt = y_samples
        x_samples_opt_torch = x_samples_torch
        y_samples_opt_torch = y_samples_torch
    
    elif method == 'relevance_pursuit':
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for relevance pursuit method")
                
        # Convert to double precision for BoTorch (recommended for better precision/stability)
        x_samples_torch_double = x_samples_torch.double()
        y_samples_torch_double = y_samples_torch.double()
        
        # Check for NaN/Inf values in the data
        if torch.any(torch.isnan(x_samples_torch_double)) or torch.any(torch.isinf(x_samples_torch_double)):
            print("Warning: NaN or Inf detected in X samples. Replacing with zeros.")
            x_samples_torch_double = torch.nan_to_num(x_samples_torch_double, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.any(torch.isnan(y_samples_torch_double)) or torch.any(torch.isinf(y_samples_torch_double)):
            print("Warning: NaN or Inf detected in y samples. Replacing with zeros.")
            y_samples_torch_double = torch.nan_to_num(y_samples_torch_double, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create a fresh kernel for relevance pursuit (don't use fixed kernel parameters)
        # Relevance pursuit needs to learn kernel hyperparameters while detecting outliers
        if n_dims == 1:
            kernel_rp = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ).double()
        else:
            kernel_rp = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            ).double()
        
        # Create BoTorch RobustRelevancePursuitSingleTaskGP model
        # Note: This model automatically handles outlier detection
        n_train = len(x_samples_torch_double)
        prior_mean_of_support = int(0.2 * n_train)
        
        rp_model = RobustRelevancePursuitSingleTaskGP(
            train_X=x_samples_torch_double,
            train_Y=y_samples_torch_double.unsqueeze(-1) if y_samples_torch_double.ndim == 1 else y_samples_torch_double,
            covar_module=kernel_rp,
            convex_parameterization=True,
            cache_model_trace=False,
            prior_mean_of_support=prior_mean_of_support
        )
        
        # Create marginal log likelihood
        mll = ExactMarginalLogLikelihood(likelihood=rp_model.likelihood, model=rp_model)
        
        # Fit using relevance pursuit (this automatically detects and handles outliers)
        print("Running relevance pursuit algorithm...")
        numbers_of_outliers = [0, int(0.05*n_train), int(0.1*n_train), int(0.15*n_train), 
                              int(0.2*n_train), int(0.3*n_train), int(0.4*n_train), 
                              int(0.5*n_train), int(0.75*n_train), n_train]
        rp_kwargs = {
            "numbers_of_outliers": numbers_of_outliers,
            "optimizer_kwargs": {"options": {"maxiter": 1024}},
        }
        fit_gpytorch_mll(mll, **rp_kwargs)
        
        # Use the fitted RP model directly - it handles outliers internally
        # No need to filter samples; the model learns which points are outliers
        x_samples_opt = x_samples
        y_samples_opt = y_samples
        x_samples_opt_torch = x_samples_torch
        y_samples_opt_torch = y_samples_torch
        
        # Use the fitted RP model for predictions
        model = rp_model
        likelihood = rp_model.likelihood
    
    elif method == 'warped_gp':
        if not BOTORCH_AVAILABLE:
            raise ImportError("BoTorch is required for warped GP method")
                
        # Convert to double precision for BoTorch (recommended for better precision/stability)
        x_samples_torch_double = x_samples_torch.double()
        y_samples_torch_double = y_samples_torch.double()
        
        # Check for NaN/Inf values in the data (can happen with constant features after normalization)
        if torch.any(torch.isnan(x_samples_torch_double)) or torch.any(torch.isinf(x_samples_torch_double)):
            print("Warning: NaN or Inf detected in X samples. Replacing with zeros.")
            x_samples_torch_double = torch.nan_to_num(x_samples_torch_double, nan=0.0, posinf=0.0, neginf=0.0)
        
        if torch.any(torch.isnan(y_samples_torch_double)) or torch.any(torch.isinf(y_samples_torch_double)):
            print("Warning: NaN or Inf detected in y samples. Replacing with zeros.")
            y_samples_torch_double = torch.nan_to_num(y_samples_torch_double, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create a fresh kernel for warped GP (don't use fixed kernel parameters)
        # Warped GP needs to learn kernel hyperparameters in the warped space
        if n_dims == 1:
            kernel_warped = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            ).double()
        else:
            kernel_warped = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=n_dims)
            ).double()
        
        # Determine bounds for input warping
        # Bounds should be of shape (2, d) where d is the number of dimensions
        if n_dims == 1:
            # For 1D, ensure we get tensors not scalars
            x_min = x_samples_torch_double.min().unsqueeze(0)
            x_max = x_samples_torch_double.max().unsqueeze(0)
        else:
            x_min = x_samples_torch_double.min(dim=0)[0]
            x_max = x_samples_torch_double.max(dim=0)[0]
        
        # Stack to create (2, d) shape: [lower_bounds, upper_bounds]
        bounds_warp = torch.stack([x_min, x_max])
        
        # Initialize Warp input transformation
        # Use LogNormalPrior with median at 1 (when a=1 and b=1, Kumaraswamy CDF is identity)
        warp_tf = Warp(
            d=n_dims,
            indices=list(range(n_dims)),
            concentration1_prior=LogNormalPrior(0.0, 0.75**0.5),
            concentration0_prior=LogNormalPrior(0.0, 0.75**0.5),
            bounds=bounds_warp
        )
        
        # Create noise variance tensor (known noise)
        train_yvar = torch.full_like(y_samples_torch_double, white_noise_level if white_noise_level is not None else gp_alpha).double()
        
        
        # Create BoTorch SingleTaskGP with input warping
        try:
            warped_model = SingleTaskGP(
                train_X=x_samples_torch_double,
                train_Y=y_samples_torch_double.unsqueeze(-1) if y_samples_torch_double.ndim == 1 else y_samples_torch_double,
                train_Yvar=train_yvar.unsqueeze(-1) if train_yvar.ndim == 1 else train_yvar,
                covar_module=kernel_warped,
                input_transform=warp_tf
            )

            # Create marginal log likelihood
            mll = ExactMarginalLogLikelihood(likelihood=warped_model.likelihood, model=warped_model)
            
            # Fit the warped GP (learns warping parameters and GP hyperparameters jointly)
            fit_gpytorch_mll(mll)

        except Exception as _:
            warped_model = SingleTaskGP(
                train_X=x_samples_torch_double,
                train_Y=y_samples_torch_double.unsqueeze(-1) if y_samples_torch_double.ndim == 1 else y_samples_torch_double,
                covar_module=kernel_warped,
                )
            
            # Create marginal log likelihood
            mll = ExactMarginalLogLikelihood(likelihood=warped_model.likelihood, model=warped_model)
            # Fit the warped GP (learns warping parameters and GP hyperparameters jointly)
            fit_gpytorch_mll(mll)
        
        
        
        # Use the fitted warped GP model directly
        x_samples_opt = x_samples
        y_samples_opt = y_samples
        x_samples_opt_torch = x_samples_torch
        y_samples_opt_torch = y_samples_torch
                
        # Use the fitted warped GP model for predictions
        model = warped_model
        likelihood = warped_model.likelihood
    
    elif method == 'heteroscedastic_gp':
        # Most Likely Heteroscedastic GP from the paper:
        # "Most Likely Heteroscedastic Gaussian Process Regression"
        
        # Create kernel for the GP
        mean_kernel, _ = create_kernel_from_config(kernel_config, n_dims, fixed_kernel, device)
        
        # Create heteroscedastic GP model using the Most Likely approach
        hetero_model = HeteroscedasticGPModel(
            train_x=x_samples_torch,
            train_y=y_samples_torch,
            kernel=mean_kernel,
            max_iter=10,
            tol=1e-03,
            var_estimate='paper',
            var_samples=1000,
            norm_and_std=True
        )
        
        # Fit the model using the Most Likely algorithm
        hetero_model.fit()
        
        # No sample optimization needed - the model learns heteroscedastic noise
        x_samples_opt = x_samples
        y_samples_opt = y_samples
        x_samples_opt_torch = x_samples_torch
        y_samples_opt_torch = y_samples_torch
        
        # Store the heteroscedastic model
        model = hetero_model
        likelihood = None  # Not used for this model
        
    
    elif method == 'sequential':
        x_samples_opt_torch, y_samples_opt_torch, _, curvature_history = sequentially_optimize_samples(
            model,
            likelihood,
            x_samples_torch.clone(),
            y_samples_torch.clone(),
            x_A_torch,
            max_iter=len(x_samples) - 2,
            noise_type=noise,
            show_hessian=show_hessian,
            calculate_convexity=calculate_convexity,
            A_limits=A_limits
        )
        x_samples_opt = x_samples_opt_torch.cpu().numpy()
        y_samples_opt = y_samples_opt_torch.cpu().numpy()
    
    elif method == 'batch':
        x_samples_opt_torch, y_samples_opt_torch = batch_optimize_samples(
            model,
            x_samples_torch.clone(),
            y_samples_torch.clone(),
            x_A_torch,
            noise_type=noise
        )
        x_samples_opt = x_samples_opt_torch.cpu().numpy()
        y_samples_opt = y_samples_opt_torch.cpu().numpy()
    
    elif method == 'targetSampling':
        x_samples_opt, y_samples_opt = filter_samples(x_samples, y_samples, A_limits)
        x_samples_opt_torch = torch.from_numpy(x_samples_opt).float().to(device)
        y_samples_opt_torch = torch.from_numpy(y_samples_opt).float().to(device)
    
    else:
        raise ValueError("Invalid method. Choose from 'none', 'sequential', 'batch', 'targetSampling', 'relevance_pursuit', 'warped_gp', 'heteroscedastic_gp'.")
    
    # Refit model with optimized samples (skip for BoTorch models as they're already fitted)
    if method not in ['relevance_pursuit', 'warped_gp', 'heteroscedastic_gp']:
        try:
            model.set_train_data(x_samples_opt_torch, y_samples_opt_torch, strict=False)
            model.train()
            likelihood.train()
            
            # Optimize hyperparameters
            # When fixed_kernel=True: only optimize likelihood (noise)
            # When fixed_kernel=False: optimize both kernel and likelihood
            if fixed_kernel:
                # Only optimize likelihood parameters (noise)
                optimizer = torch.optim.Adam(likelihood.parameters(), lr=0.1)
            else:
                # Optimize both model (kernel) and likelihood parameters
                # Note: model.parameters() already includes likelihood parameters
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
            
            training_iter = 50
            for i in range(training_iter):
                optimizer.zero_grad()
                output = model(x_samples_opt_torch)
                loss = -mll(output, y_samples_opt_torch)
                loss.backward()
                optimizer.step()
            
            # Make predictions
            model.eval()
            likelihood.eval()
        
        except Exception as e:
            print(f"GP fitting or prediction failed: {e}")
            import traceback
            traceback.print_exc()
            y_pred_opt = np.zeros_like(y)
            y_std_opt = np.ones_like(y) * 5
            errors = calculate_prediction_errors(
                x, y_pred_opt, y_std_opt, y, A_limits, 
                plot=show_plots, 
                title=f"{method}{noise} Calibration Curve {seed}"
            )
            return errors, len(x_samples_opt)
    
    # Make predictions (for all methods including BoTorch models)
    try:
        model.eval()
        likelihood.eval() if not method == 'heteroscedastic_gp' else None
        
        x_torch = torch.from_numpy(x).float().to(device)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # For BoTorch models (relevance_pursuit, warped_gp), use the posterior method
            if method in ['relevance_pursuit', 'warped_gp']:
                # Convert to double for BoTorch
                x_torch_double = x_torch.double()
                
                # Suppress expected warning about robust rho not being applied to test data
                # This is correct behavior - rho is only defined for training points
                if method == 'relevance_pursuit':
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='.*Robust rho not applied.*')
                        # observation_noise=True includes the likelihood noise in predictions
                        posterior = model.posterior(x_torch_double, observation_noise=True)
                else:
                    posterior = model.posterior(x_torch_double, observation_noise=True)
                
                y_pred_opt = posterior.mean.squeeze(-1).cpu().numpy()
                y_std_opt = posterior.variance.squeeze(-1).sqrt().cpu().numpy()
            
            elif method == 'heteroscedastic_gp':
                # For heteroscedastic GP, use the posterior with observation noise
                # This includes the learned heteroscedastic noise
                posterior = model.posterior(x_torch, observation_noise=True)
                
                y_pred_opt = posterior.mean.squeeze(-1).cpu().numpy()
                y_std_opt = posterior.variance.squeeze(-1).sqrt().cpu().numpy()
            
            else:
                # For GPyTorch models, use likelihood(model(x))
                predictions = likelihood(model(x_torch))
                y_pred_opt = predictions.mean.cpu().numpy()
                y_std_opt = predictions.stddev.cpu().numpy()
    
    except Exception as e:
        print(f"GP fitting or prediction failed: {e}")
        import traceback
        traceback.print_exc()
        y_pred_opt = np.zeros_like(y)
        y_std_opt = np.ones_like(y) * 5
    
    errors = calculate_prediction_errors(
        x, y_pred_opt, y_std_opt, y, A_limits, 
        plot=show_plots, 
        title=f"{method}{noise} Calibration Curve {seed}"
    )
    
    # Add runtime to errors
    runtime = time.time() - start_time
    errors['Runtime'] = runtime
    
    # Only plot for 1D case (single method plots)
    if show_plots and n_dims == 1 and plot_func is not None and not return_plot_data:
        plot_predictions(
            plot_func,
            x.flatten(),
            x_samples_opt.flatten(),
            y_samples_opt,
            y_pred_opt,
            y_std_opt,
            X_limits,
            A_limits,
            title=f"{method}{noise} Optimized GP Predictions {seed}"
        )
    
    # Plot curvature bounds if they were calculated
    if calculate_convexity and show_plots and curvature_history and curvature_history.get('m_epi'):
        plot_curvature_bounds(
            curvature_history, 
            title="Curvature Bound"
        )
    
    # Return plot data if requested (for comparison plots)
    if return_plot_data:
        plot_data = {
            'x': x.flatten() if n_dims == 1 else x,
            'x_samples': x_samples_opt.flatten() if n_dims == 1 else x_samples_opt,
            'y_samples': y_samples_opt,
            'y_pred': y_pred_opt,
            'y_std': y_std_opt,
            'objective_func': plot_func
        }
        return errors, len(x_samples_opt), plot_data
    
    return errors, len(x_samples_opt)