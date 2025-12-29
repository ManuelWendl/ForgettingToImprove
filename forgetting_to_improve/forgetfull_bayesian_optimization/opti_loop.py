import torch
from botorch import fit_gpytorch_mll
from botorch.utils.sampling import draw_sobol_samples # type: ignore
from .opti_aquisition import get_optimize_acqf_and_get_observation
from .target_region import TargetRegion
from .filter_samples import filter_samples

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
        num_target_region_samples = opti_options.get('num_target_region_samples', 1000)
    else:
        algorithm = None
        method = None


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

        mll, model = initialize_model(train_x, train_obj, global_bounds)

        if algorithm:
            print(f"Using optimization algorithm: {algorithm} with method: {method}")
        
        for i in range(1, num_iters + 1):
            if algorithm == 'joint' or algorithm == 'epistemic':
                target_region = TargetRegion(global_bounds, num_initial_points=num_target_region_samples)
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
                mll, model = initialize_model(filtered_x_samples, filtered_y_samples, global_bounds)
            try:
                fit_gpytorch_mll(mll)
            except Exception as e:
                print(f"Error fitting model at iteration {i}: {e}")
            aq = aq_func(model=model, X_baseline=train_x)

            if algorithm == 'joint' or algorithm == 'epistemic':
                # update target region
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
                new_x, new_acqvalue = optimize_acqf_and_get_observation(aq, global_bounds)

            exact_obj = obj(new_x).unsqueeze(-1)
            new_obj = exact_obj + obs_noise * torch.randn_like(exact_obj)
            train_x = torch.cat([train_x, new_x])
            train_obj = torch.cat([train_obj, new_obj])

            # update progress
            best_value = obj(train_x).max().item()
            best_observed.append(best_value)

            mll, model = initialize_model(train_x, train_obj, global_bounds)
            print(f"Iteration {i}: Best observed value = {best_value}")
        return train_x, train_obj, best_observed

    return optimization_loop