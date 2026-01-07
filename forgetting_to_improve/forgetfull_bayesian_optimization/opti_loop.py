import torch
import warnings
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from .opti_aquisition import get_optimize_acqf_and_get_observation
from .target_region import TargetRegion
from .filter_samples import filter_samples
from .baseline_models import (
    initialize_relevance_pursuit_model,
    initialize_warped_gp_model,
    initialize_heteroscedastic_gp_model,
    initialize_modulating_surrogates_model
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
        optimize_acqf_and_get_observation = get_optimize_acqf_and_get_observation(
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            obj=obj,
            obs_noise=obs_noise,
        )

        train_x = draw_sobol_samples(
            bounds=global_bounds, n=n_initial_samples, q=1, seed=seed
        ).squeeze(1)
        exact_obj = obj(train_x).unsqueeze(-1)  # add output dimension

        best_observed_value = exact_obj.max().item()
        best_observed = [best_observed_value]
        train_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)

        # Initial model
        mll, model = initialize_model(train_x, train_obj, global_bounds)

        if algorithm:
            print(f"Using optimization algorithm: {algorithm}" + 
                  (f" with method: {method}" if not is_baseline_method else " (baseline)"))
        
        for i in range(1, num_iters + 1):
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
                    global_bounds=global_bounds
                )
            
            elif algorithm == 'heteroscedastic_gp':
                # Create fresh kernel for each iteration
                import gpytorch
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
                target_region = TargetRegion(global_bounds, num_initial_points=num_target_region_samples, seed=seed, iteration=i)
                
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
                    global_bounds=global_bounds,
                )
                
                # Reinitialize model with filtered samples
                mll, model = initialize_model(filtered_x_samples, filtered_y_samples, global_bounds)
            
            else:
                # No special handling (algorithm == 'none')
                mll, model = initialize_model(train_x, train_obj, global_bounds)
            
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
                # For modulating surrogates, augment X_baseline with random modulating values
                n_train = train_x.shape[0]
                modulating_dims = torch.rand(n_train, 1, dtype=train_x.dtype, device=train_x.device)
                train_x_augmented = torch.cat([train_x, modulating_dims], dim=-1)
                aq = aq_func(model=model, X_baseline=train_x_augmented)
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
                # For modulating surrogates, optimize over augmented space
                # Augment global bounds with [0, 1] for the modulating dimension
                augmented_bounds = torch.cat([
                    global_bounds,
                    torch.tensor([[0.0], [1.0]], dtype=global_bounds.dtype, device=global_bounds.device)
                ], dim=1)
                new_x_aug, new_acqvalue = optimize_acqf_and_get_observation(aq, augmented_bounds)
                # Remove the modulating dimension from the candidate for objective evaluation
                new_x = new_x_aug[:, :-1]
            else:
                # Use global bounds for baseline methods or no algorithm
                new_x, new_acqvalue = optimize_acqf_and_get_observation(aq, global_bounds)

            # Evaluate objective and add to training data
            exact_obj = obj(new_x).unsqueeze(-1)
            new_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # update progress
            best_value = obj(train_x).max().item()
            best_observed.append(best_value)
            
            print(f"Iteration {i}: Best observed value = {best_value}")
        
        return train_x, train_obj, best_observed

    return optimization_loop