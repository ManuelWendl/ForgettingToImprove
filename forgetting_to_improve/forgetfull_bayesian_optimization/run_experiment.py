from .opti_loop import get_optimization_loop

def run_experiment(
        aq_func, 
        obj, 
        initialize_model, 
        num_restarts, 
        raw_samples, 
        obs_noise, 
        n_initial_samples, 
        global_bounds,
        num_iters,
        num_seeds,
        opti_options=None,
    ):
    """Runs a full Bayesian optimization experiment."""
    best_observed_all = []
    iteration_times_all = []
    for seed in range(num_seeds):
        optimization_loop = get_optimization_loop(
            aq_func=aq_func,
            obj=obj,
            initialize_model=initialize_model,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            obs_noise=obs_noise,
            n_initial_samples=n_initial_samples,
            global_bounds=global_bounds,
            opti_options=opti_options,
        )
        
        _, _, best_observed, iteration_times = optimization_loop(num_iters, seed)
        best_observed_all.append(best_observed)
        iteration_times_all.append(iteration_times)
    
    return best_observed_all, iteration_times_all
    