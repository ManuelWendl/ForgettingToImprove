import torch
import warnings
import gpytorch
import time
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from .opti_aquisition import get_optimize_acqf_and_get_observation
from .target_region import TargetRegion
from .filter_samples import filter_samples
from .baseline_models import (
    initialize_relevance_pursuit_model,
    initialize_warped_gp_model,
    initialize_heteroscedastic_gp_model,
    initialize_modulating_surrogates_model,
    create_mc_latent_acquisition
)

def get_optimization_loop(
        aq_func, 
        obj, 
        initialize_model, 
        num_restarts, 
        raw_samples, 
        obs_noise, 
        n_initial_samples, 
        global_bounds,
        opti_options=None,
        ):
    """Returns an optimization loop function for Bayesian optimization."""
    if opti_options and hasattr(opti_options, 'get'):
        algorithm = opti_options.get('algorithm', 'joint')
        method = opti_options.get('method', 'sequential')
        min_samples = opti_options.get('min_samples', 5)
        num_target_region_samples = opti_options.get('num_target_region_samples', 10000)
    else:
        algorithm = None
        method = None

    # Check if using baseline method
    is_baseline_method = algorithm in ['relevance_pursuit', 'warped_gp', 'heteroscedastic_gp', 'modulating_surrogates']

    def optimization_loop(num_iters, seed):
        """Runs the Bayesian optimization loop."""
        # Normalize bounds to [0, 1] hypercube for consistent GP priors
        # Store original bounds for denormalization
        original_bounds = global_bounds
        normalized_bounds = torch.stack([
            torch.zeros(global_bounds.shape[1], dtype=global_bounds.dtype, device=global_bounds.device),
            torch.ones(global_bounds.shape[1], dtype=global_bounds.dtype, device=global_bounds.device)
        ])
        
        # Wrap objective function to handle denormalization
        def normalized_obj(x_norm):
            """Evaluate objective at normalized inputs by denormalizing first."""
            x_original = unnormalize(x_norm, original_bounds)
            return obj(x_original)
        
        optimize_acqf_and_get_observation = get_optimize_acqf_and_get_observation(
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            obj=normalized_obj,  # Use wrapped objective
            obs_noise=obs_noise,
        )

        # Sample in normalized [0,1] space
        train_x = draw_sobol_samples(
            bounds=normalized_bounds, n=n_initial_samples, q=1, seed=seed
        ).squeeze(1)
        exact_obj = normalized_obj(train_x).unsqueeze(-1)  # add output dimension

        best_observed_value = exact_obj.max().item()
        best_observed = [best_observed_value]
        train_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)

        # Initial model - pass normalized bounds
        mll, model = initialize_model(train_x, train_obj, normalized_bounds)

        if algorithm:
            print(f"Using optimization algorithm: {algorithm}" + 
                  (f" with method: {method}" if not is_baseline_method else " (baseline)"))
        
        iteration_times = []
        for i in range(1, num_iters + 1):
            iteration_start_time = time.time()
            # Handle baseline methods
            if algorithm == 'relevance_pursuit':
                # Reinitialize with relevance pursuit
                mll, model = initialize_relevance_pursuit_model(
                    train_x=train_x,
                    train_y=train_obj,
                    device=train_x.device
                )
            
            elif algorithm == 'warped_gp':
                # Reinitialize with warped GP - pass None for kernel to create fresh one with same priors
                noise_level = obs_noise if obs_noise > 0 else 0.1
                mll, model = initialize_warped_gp_model(
                    train_x=train_x,
                    train_y=train_obj,
                    kernel=None,  # Create fresh kernel, don't use pre-optimized one
                    noise_level=noise_level,
                    device=train_x.device,
                    global_bounds=normalized_bounds
                )
            
            elif algorithm == 'heteroscedastic_gp':
                # Create fresh kernel for each iteration
                n_dims = train_x.shape[-1]
                fresh_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dims)
                )
                # Reinitialize with heteroscedastic GP
                mll, model = initialize_heteroscedastic_gp_model(
                    train_x=train_x,
                    train_y=train_obj,
                    mean_kernel=fresh_kernel,
                    device=train_x.device
                )
            
            elif algorithm == 'modulating_surrogates':
                # Reinitialize with modulating surrogates GP
                noise_level = obs_noise if obs_noise > 0 else 0.1
                mll, model = initialize_modulating_surrogates_model(
                    train_x=train_x,
                    train_y=train_obj,
                    kernel=None,
                    noise_level=noise_level,
                    device=train_x.device
                )
            
            # Handle forgetting-based methods
            elif algorithm in ['joint', 'epistemic']:
                target_region = TargetRegion(normalized_bounds, num_initial_points=num_target_region_samples, seed=seed, iteration=i)
                
                # Filter samples
                filtered_x_samples, filtered_y_samples, _ = filter_samples(
                    model=model,
                    mll=mll,
                    train_x=train_x.clone(),
                    train_obj=train_obj.clone(),
                    target_region=target_region,
                    algorithm=algorithm,
                    method=method,
                    min_samples=min_samples,
                    initialize_model=initialize_model,
                    global_bounds=normalized_bounds,
                )
                
                # Reinitialize model with filtered samples
                mll, model = initialize_model(filtered_x_samples, filtered_y_samples, normalized_bounds)
            
            else:
                # No special handling (algorithm == 'none')
                mll, model = initialize_model(train_x, train_obj, normalized_bounds)
            
            # Fit model (skip for heteroscedastic as it's already fitted)
            if algorithm != 'heteroscedastic_gp':
                try:
                    # Suppress warnings for relevance pursuit
                    if algorithm == 'relevance_pursuit':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', message='.*Robust rho not applied.*')
                            if mll is not None:
                                fit_gpytorch_mll(mll)
                    else:
                        if mll is not None:
                            fit_gpytorch_mll(mll)
                except Exception as e:
                    print(f"Error fitting model at iteration {i}: {e}")
            
            # Create acquisition function
            if algorithm == 'modulating_surrogates':
                # For modulating surrogates (Latent GP), use Monte Carlo acquisition
                # Average acquisition function over posterior samples of latent variables
                try:
                    aq = create_mc_latent_acquisition(
                        acq_func_factory=aq_func,
                        latent_gp_model=model,
                        X_baseline=train_x,  # Original dimension, no augmentation needed
                        num_samples=None  # Use all MCMC samples
                    )
                except Exception as e:
                    print(f"Warning: Failed to create MC acquisition ({e}), using standard acquisition")
                    aq = aq_func(model=model, X_baseline=train_x)
            else:
                aq = aq_func(model=model, X_baseline=train_x)

            # Optimize acquisition function
            if algorithm in ['joint', 'epistemic']:
                # Update target region and optimize within bounds
                target_region.update(model)
                bounds = target_region.get_bounds()
                new_x = None
                new_acqvalue = -torch.inf
            
                for b in bounds:
                    # optimize within each bounding box
                    bound = torch.stack(b, dim=0)
                    candidate_x, candidate_acqvalue = optimize_acqf_and_get_observation(aq, bound)
                    if candidate_acqvalue > new_acqvalue:
                        new_x = candidate_x
                        new_acqvalue = candidate_acqvalue
            elif algorithm == 'modulating_surrogates':
                # For modulating surrogates (Latent GP), optimize in normalized space
                # The latent variables are marginalized out via Monte Carlo in the acquisition function
                new_x, new_acqvalue = optimize_acqf_and_get_observation(aq, normalized_bounds)
            else:
                # Use normalized bounds for baseline methods or no algorithm
                new_x, new_acqvalue = optimize_acqf_and_get_observation(aq, normalized_bounds)

            # Evaluate objective and add to training data (new_x is in normalized space)
            exact_obj = normalized_obj(new_x).unsqueeze(-1)
            new_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # update progress - evaluate at denormalized points
            train_x_original = unnormalize(train_x, original_bounds)
            best_value = obj(train_x_original).max().item()
            best_observed.append(best_value)
            
            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)
            
            print(f"Iteration {i}: Best observed value = {best_value}")
        
        return train_x, train_obj, best_observed, iteration_times

    return optimization_loop