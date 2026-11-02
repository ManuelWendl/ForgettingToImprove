import torch
import warnings
import gpytorch
import time
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import unnormalize, standardize
from .opti_aquisition import get_optimize_acqf_and_get_observation
from .target_region import TargetRegion
from .filter_samples import filter_samples
from .baseline_models import (
    initialize_relevance_pursuit_model,
    initialize_warped_gp_model,
    initialize_trimmed_mll_model,
    initialize_student_t_gp_model,
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
        perturbation_config=None,
        ):
    """Returns an optimization loop function for Bayesian optimization."""
    if opti_options and hasattr(opti_options, 'get'):
        algorithm = opti_options.get('algorithm', 'joint')
        method = opti_options.get('method', 'sequential')
        min_samples = opti_options.get('min_samples', 5)
        num_target_region_samples = opti_options.get('num_target_region_samples', 10000)
        rolling_window = opti_options.get('rolling_window', None)
    else:
        algorithm = None
        method = None
        rolling_window = None

    # Check if using baseline method
    is_baseline_method = algorithm in ['relevance_pursuit', 'warped_gp', 'trimmed_mll', 'student_t']
    # AutoML CSV benchmark only: start each new task from fresh samples
    reset_per_task = opti_options.get('reset_per_task', False) if opti_options else False

    # Setup perturbation if configured
    apply_perturbation = False
    perturbation_type = None
    perturbation_prob = 0.0
    perturbation_value = 0.0
    
    if perturbation_config:
        perturbation_type = perturbation_config.get('type', None)
        perturbation_prob = perturbation_config.get('probability', 0.0)
        perturbation_value = perturbation_config.get('value', 0.0)
        apply_perturbation = perturbation_type is not None and perturbation_prob > 0.0
        
        if apply_perturbation:
            print(f"Perturbation enabled: type={perturbation_type}, prob={perturbation_prob}, value={perturbation_value}")

    def optimization_loop(num_iters, seed):
        """Runs the Bayesian optimization loop."""
        # Seed torch's global RNG (used e.g. by the NN-kernel MLP initialization)
        # so that every (method, seed) run is reproducible.
        torch.manual_seed(seed)

        # Normalize bounds to [0, 1] hypercube for consistent GP priors
        # Store original bounds for denormalization
        original_bounds = global_bounds
        normalized_bounds = torch.stack([
            torch.zeros(global_bounds.shape[1], dtype=global_bounds.dtype, device=global_bounds.device),
            torch.ones(global_bounds.shape[1], dtype=global_bounds.dtype, device=global_bounds.device)
        ])
        
        # Wrap objective function to handle denormalization and perturbation
        def normalized_obj(x_norm):
            """Evaluate objective at normalized inputs by denormalizing first."""
            x_original = unnormalize(x_norm, original_bounds)
            obj_values = obj(x_original)
            
            # Apply perturbation if configured
            if apply_perturbation:
                # Generate random values for each evaluation
                batch_size = obj_values.shape[0] if obj_values.dim() > 0 else 1
                corruption_mask = torch.rand(batch_size) < perturbation_prob
                
                if corruption_mask.any():
                    if perturbation_type == 'constant':
                        # Apply constant value perturbation
                        obj_values[corruption_mask] = perturbation_value
            
            return obj_values
        
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

        # AutoML CSV benchmark: select the task of iteration 0
        if hasattr(obj, 'set_iteration'):
            obj.set_iteration(0)

        exact_obj = normalized_obj(train_x).unsqueeze(-1)  # add output dimension

        best_observed_value = exact_obj.max().item()
        best_observed = [best_observed_value]
        train_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)

        # Current task (AutoML CSV benchmark only) and running best on it
        current_task = getattr(obj, 'current_task', None)
        running_best = best_observed_value
        previous_model_state = None

        # Initial model - pass normalized bounds
        mll, model = initialize_model(train_x, train_obj, normalized_bounds)

        if algorithm:
            print(f"Using optimization algorithm: {algorithm}" + 
                  (f" with method: {method}" if not is_baseline_method else " (baseline)"))
        
        iteration_times = []
        for i in range(1, num_iters + 1):
            iteration_start_time = time.time()

            # AutoML CSV benchmark: switch to the task of this iteration
            if hasattr(obj, 'set_iteration'):
                obj.set_iteration(i)
                # Check if task changed - reset running best if so
                new_task = getattr(obj, 'current_task', None)
                if new_task is not None and new_task != current_task:
                    current_task = new_task
                    running_best = float('-inf')  # Reset for new task
                    print(f"Task changed to {current_task}, resetting running best")

                    if reset_per_task:
                        # Keep current kernel/hyperparameter state for reload
                        if model is not None and hasattr(model, 'state_dict'):
                            try:
                                previous_model_state = model.state_dict()
                            except Exception:
                                previous_model_state = None

                        # Reset training set to fresh random samples for new task
                        train_x = draw_sobol_samples(
                            bounds=normalized_bounds,
                            n=n_initial_samples,
                            q=1,
                            seed=seed + i,
                        ).squeeze(1)

                        exact_obj = normalized_obj(train_x).unsqueeze(-1)
                        train_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)
                        running_best = exact_obj.max().item()

            # Handle baseline methods
            if algorithm == 'relevance_pursuit':
                # Create fresh kernel for each iteration
                n_dims = train_x.shape[-1]
                fresh_kernel = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=n_dims)
                )
                # Reinitialize with relevance pursuit
                mll, model = initialize_relevance_pursuit_model(
                    train_x=train_x,
                    train_y=standardize(train_obj),
                    kernel=fresh_kernel,
                    device=train_x.device
                )
            
            elif algorithm == 'warped_gp':
                # Reinitialize with warped GP (fresh kernel)
                noise_level = obs_noise if obs_noise > 0 else 0.1
                mll, model = initialize_warped_gp_model(
                    train_x=train_x,
                    train_y=standardize(train_obj),
                    kernel=None,
                    noise_level=noise_level,
                    device=train_x.device,
                    global_bounds=normalized_bounds
                )
            
            elif algorithm == 'trimmed_mll':
                # Reinitialize with trimmed MLL GP
                noise_level = obs_noise if obs_noise > 0 else 0.1
                mll, model = initialize_trimmed_mll_model(
                    train_x=train_x,
                    train_y=standardize(train_obj),
                    kernel=None,
                    noise_level=noise_level,
                    device=train_x.device,
                    max_outlier_fraction=0.5,
                    max_iterations=50
                )

            elif algorithm == 'student_t':
                # Reinitialize with Student-t GP
                noise_level = obs_noise if obs_noise > 0 else 0.1
                mll, model = initialize_student_t_gp_model(
                    train_x=train_x,
                    train_y=standardize(train_obj),
                    kernel=None,
                    noise_level=noise_level,
                    device=train_x.device,
                    degrees_of_freedom=3.0
                )

            # Handle forgetting-based methods
            elif algorithm in ['joint', 'epistemic']:
                target_region = TargetRegion(normalized_bounds, num_initial_points=num_target_region_samples, seed=seed, iteration=i)
                
                # Filter samples
                filtered_x_samples, filtered_y_samples, _ = filter_samples(
                    model=model,
                    mll=mll,
                    train_x=train_x.clone(),
                    train_obj=standardize(train_obj.clone()),
                    target_region=target_region,
                    algorithm=algorithm,
                    method=method,
                    min_samples=min_samples,
                    initialize_model=initialize_model,
                    global_bounds=normalized_bounds,
                )
                
                # Reinitialize model with filtered samples
                mll, model = initialize_model(filtered_x_samples, standardize(filtered_y_samples), normalized_bounds)

            # Target region sampling baseline: only refit the GP on the
            # observed points that fall inside the current target region,
            # rather than filtering by per-sample influence like 'joint'/'epistemic'.
            elif algorithm == 'target_region':
                target_region = TargetRegion(normalized_bounds, num_initial_points=num_target_region_samples, seed=seed, iteration=i)
                if target_region.samples.shape[0] > 2:
                    target_region.update(model)
                region_bounds = target_region.get_bounds()

                in_region_mask = torch.zeros(train_x.shape[0], dtype=torch.bool)
                for lo, hi in region_bounds:
                    in_box = ((train_x >= lo) & (train_x <= hi)).all(dim=-1)
                    in_region_mask = in_region_mask | in_box

                if in_region_mask.sum().item() < min_samples:
                    # Too few observed points fall inside the region to fit a
                    # meaningful GP -- fall back to the min_samples points
                    # closest to the region instead of an under-determined fit.
                    region_centers = torch.stack([(lo + hi) / 2.0 for lo, hi in region_bounds])
                    dists = torch.cdist(train_x, region_centers).min(dim=-1).values
                    n_fallback = min(min_samples, train_x.shape[0])
                    closest_indices = torch.topk(dists, n_fallback, largest=False).indices
                    in_region_mask = torch.zeros(train_x.shape[0], dtype=torch.bool)
                    in_region_mask[closest_indices] = True

                filtered_x_samples = train_x[in_region_mask]
                filtered_y_samples = train_obj[in_region_mask]

                # Reinitialize model with target-region-filtered samples
                mll, model = initialize_model(filtered_x_samples, standardize(filtered_y_samples), normalized_bounds)

            else:
                # No special handling (algorithm == 'none')
                mll, model = initialize_model(train_x, standardize(train_obj), normalized_bounds)

            # After a task reset, reload the previous kernel/hyperparameters
            # (only the samples are dropped)
            if reset_per_task and previous_model_state is not None and model is not None:
                if hasattr(model, 'load_state_dict'):
                    try:
                        model.load_state_dict(previous_model_state, strict=False)
                    except Exception as e:
                        print(f"Warning: could not load previous model state after reset: {e}")
                previous_model_state = None

            # Fit model
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
            else:
                # Use normalized bounds for baseline methods or no algorithm
                new_x, new_acqvalue = optimize_acqf_and_get_observation(aq, normalized_bounds)

            # Evaluate objective and add to training data (new_x is in normalized space)
            exact_obj = normalized_obj(new_x).unsqueeze(-1)
            new_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # Keep only the most recent samples (optional, AutoML CSV benchmark)
            if rolling_window is not None and train_x.shape[0] > rolling_window:
                train_x = train_x[-rolling_window:]
                train_obj = train_obj[-rolling_window:]

            # Update progress (on the current task)
            new_value = exact_obj.item()
            running_best = max(running_best, new_value)
            best_observed.append(running_best)

            iteration_time = time.time() - iteration_start_time
            iteration_times.append(iteration_time)

            print(f"Iteration {i}: New value = {new_value:.6f}, Running best = {running_best:.6f}")
        
        return train_x, train_obj, best_observed, iteration_times

    return optimization_loop