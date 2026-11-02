import argparse
import sys
from .evaluate import evaluate
from .helper.config_loader import ConfigLoader
from .helper.results_writer import ResultsWriter
from .helper.plot_gp import (
    plot_predictions_comparison,
    plot_calibration_comparison,
    plot_curvature_condition_comparison,
    latex_available,
)


def run_single_experiment(config, method, objective_name, objective_func, objective_type, noise, seed, results_writer,
                           write_individual=False, return_plot_data=False,
                           gamma_prior_a=None, gamma_prior_b=None):
    """Run a single experiment with given parameters."""
    result = evaluate(
        method=method,
        objective_func=objective_func,
        noise=noise,
        A_limits=config['A_limits'],
        X_limits=config['X_limits'],
        num_samples=config['num_samples'],
        num_train_samples=config['num_train_samples'],
        num_A_samples=config['num_A_samples'],
        seed=seed,
        fixed_kernel=config['fixed_kernel'],
        calculate_convexity=config.get('calculate_convexity', False),
        removal_threshold=config.get('removal_threshold', 0.0),
        map_noise=config.get('map_noise', False),
        noise_init=config.get('noise_init', 'config'),
        objective_type=objective_type,
        feature_subset=config.get('feature_subset', None),
        kernel_config=config.get('kernel_config', None),
        gp_alpha=config.get('gp_alpha', 0.01),
        gamma_prior_a=gamma_prior_a,
        gamma_prior_b=gamma_prior_b,
        return_plot_data=return_plot_data
    )
    
    if return_plot_data:
        results, num_samples_opt, plot_data = result
    else:
        results, num_samples_opt = result
        plot_data = None
    
    # Write results for single experiment only if requested
    if write_individual:
        results_writer.write_single_experiment_results(
            config, results, method, objective_name, noise, seed
        )
    
    if return_plot_data:
        return results, num_samples_opt, plot_data
    return results, num_samples_opt

def run_experiments_from_config(config_path, collect_curvature_histories=False):
    """Run experiments based on configuration file.

    Args:
        config_path: Path to the experiment YAML config.
        collect_curvature_histories: If True, also return the curvature
            histories of a gamma_prior_a sweep per objective (used by
            run_curvature_ablations.py to combine several configs into one
            figure).
    """
    config_loader = ConfigLoader()
    config = config_loader.load_config(config_path)
    results_writer = ResultsWriter()
    
    print(f"Loaded configuration for {config['type']} experiment")
    print(f"Methods: {config['methods']}")
    print(f"Objectives: {config['objectives']}")
    print(f"Noises: {config['noises']}")
    print(f"Seeds: {len(config['seeds'])} seed(s)")

    # Optional Gamma noise-prior sweep over the concentration a (curvature
    # ablation); [None] uses the default prior
    gamma_prior_a_values = config.get('gamma_prior_a_values', [None])
    gamma_prior_b_values = config.get('gamma_prior_b_values', [None])
    if len(gamma_prior_a_values) > 1:
        print(f"Gamma prior a sweep: {gamma_prior_a_values}")

    # Determine if we need to do comparisons
    compare_methods = len(config['methods']) > 1
    compare_noises = len(config['noises']) > 1
    compare_gamma_prior_a = len(gamma_prior_a_values) > 1
    is_statistical = config['type'] == 'statistic'
    is_single = config['type'] == 'single'

    # Check if we're doing a comparison plot for single experiment
    is_single_comparison = is_single and compare_methods and not compare_noises
    is_single_gamma_prior_comparison = (
        is_single and compare_gamma_prior_a and not compare_methods and not compare_noises
    )

    # Store all results for comparison
    all_results = {}
    per_seed_results = {}  # statistic experiments: variant -> seed -> metrics (+ n_opt)
    comparison_plot_data = {}  # For single experiment comparisons
    gamma_prior_curvature_histories = {}  # For single experiment a-sweep curvature overlay
    collected_curvature_histories = {}  # objective_name -> gamma_prior_curvature_histories

    # Run experiments for all combinations
    for method in config['methods']:
        for i, objective_name in enumerate(config['objectives']):
            objective_func = config['objective_funcs'][i]
            objective_type = config['objective_types'][i]
            for noise in config['noises']:
                # Skip invalid combinations
                if (method == 'targetSampling' or method == 'none' or method == 'relevance_pursuit' or method == 'warped_gp' or method == 'heteroscedastic_gp') and noise != 'joint':
                    # Skip this combination (these methods don't use the noise parameter)
                    continue

                # Create variant identifier
                if compare_methods and compare_noises:
                    variant_key = f"{method}_{noise}"
                elif compare_methods:
                    variant_key = method
                elif compare_noises:
                    variant_key = noise
                else:
                    variant_key = f"{method}_{objective_name}_{noise}"

                # Run experiments for every gamma-prior-a setting (default: single [None] -> no override)
                gamma_prior_b = gamma_prior_b_values[0] if gamma_prior_b_values else None
                for gamma_prior_a in gamma_prior_a_values:
                    variant_key_a = f"{variant_key}_a{gamma_prior_a}" if compare_gamma_prior_a else variant_key

                    # Run experiments for all seeds
                    seed_results = []
                    num_zero_samples = 0
                    for seed in config['seeds']:
                        # For method comparison, request plot data. For a
                        # gamma-prior sweep, also request it (first seed
                        # only) to collect the curvature_history overlay.
                        want_plot_data = is_single_comparison or (
                            is_single_gamma_prior_comparison and seed == config['seeds'][0]
                        )

                        if want_plot_data:
                            result = run_single_experiment(
                                config, method, objective_name, objective_func, objective_type, noise, seed, results_writer,
                                write_individual=not is_statistical,
                                return_plot_data=True,
                                gamma_prior_a=gamma_prior_a,
                                gamma_prior_b=gamma_prior_b,
                            )
                            results, num_samples_opt, plot_data = result
                            # Store plot data for the first seed only
                            if seed == config['seeds'][0] and is_single_comparison:
                                comparison_plot_data[method] = plot_data
                            if seed == config['seeds'][0] and is_single_gamma_prior_comparison:
                                ch = (plot_data or {}).get('curvature_history')
                                if ch is not None:
                                    gamma_prior_curvature_histories[f"a={gamma_prior_a}"] = ch
                        else:
                            results, num_samples_opt = run_single_experiment(
                                config, method, objective_name, objective_func, objective_type, noise, seed, results_writer,
                                write_individual=not is_statistical,
                                gamma_prior_a=gamma_prior_a,
                                gamma_prior_b=gamma_prior_b,
                            )

                        seed_results.append(results)
                        if num_samples_opt == 0:
                            num_zero_samples += 1
                        if is_statistical:
                            # n_opt: number of training samples kept
                            per_seed_results.setdefault(variant_key_a, {})[seed] = {**results, 'n_opt': num_samples_opt}

                    # Aggregate results if statistical experiment
                    if is_statistical:
                        aggregated_results = results_writer.aggregate_statistical_results(seed_results)
                        all_results[variant_key_a] = aggregated_results

                        print(f"Aggregated results for {variant_key_a}:")
                        print(f"  Mean metrics: {aggregated_results['mean']}")
                        print(f"  Min metrics: {aggregated_results['min']}")
                        print(f"  Number of Zero Samples: {num_zero_samples} out of {len(config['seeds'])} seeds")
                    else:
                        all_results[variant_key_a] = seed_results[0]

                # Create the combined curvature-condition overlay plot for a gamma-prior-a sweep
                if is_single_gamma_prior_comparison and gamma_prior_curvature_histories:
                    if collect_curvature_histories:
                        collected_curvature_histories[objective_name] = dict(gamma_prior_curvature_histories)
                    if config.get('show_plots', False):
                        plot_curvature_condition_comparison(
                            gamma_prior_curvature_histories,
                            title=f"Restricted Curvature (Gamma prior a sweep) - {objective_name}",
                        )
                    gamma_prior_curvature_histories = {}

    # Create comparison plot for single experiment with multiple methods
    if is_single_comparison and comparison_plot_data and config['show_plots']:
        print(f"Creating comparison plot for methods: {list(comparison_plot_data.keys())}")
        # Extract common data from first method
        first_method_data = next(iter(comparison_plot_data.values()))
        objective_func = first_method_data['objective_func']
        x = first_method_data['x']
        
        # Prepare data for plotting
        plot_comparison_data = {}
        for method, data in comparison_plot_data.items():
            plot_comparison_data[method] = {
                'x_samples': data['x_samples'],
                'y_samples': data['y_samples'],
                'y_pred': data['y_pred'],
                'y_std': data['y_std']
            }
        
        # Generate comparison plot
        seed = config['seeds'][0]
        objective_name = config['objectives'][0]
        usetex = latex_available()
        plot_predictions_comparison(
            objective_func,
            x,
            plot_comparison_data,
            config['X_limits'],
            config['A_limits'],
            title=f"Comparison_{objective_name}_seed{seed}",
            usetex=usetex
        )

        # Generate calibration comparison plot
        plot_calibration_comparison(
            objective_func,
            x,
            plot_comparison_data,
            title=f"Comparison_{objective_name}_seed{seed}",
            A_limits=config['A_limits'],
            usetex=usetex
        )
    
    # Write comparison results if needed
    if compare_methods or compare_noises or is_statistical:
        if compare_methods and compare_noises:
            comparison_type = "method_noise"
        elif compare_methods:
            comparison_type = "method"
        elif compare_noises:
            comparison_type = "noise"
        else:
            comparison_type = "statistical"
        
        results_writer.write_comparison_results(config, all_results, comparison_type,
                                                per_seed=per_seed_results if is_statistical else None)

    if collect_curvature_histories:
        return all_results, collected_curvature_histories
    return all_results

def main():
    """Run the experiments of a YAML config file (-c)."""
    parser = argparse.ArgumentParser(description='Run GP optimization experiments')
    parser.add_argument('--config', '-c', type=str, required=True, help='Path to configuration YAML file')
    args = parser.parse_args()
    try:
        return run_experiments_from_config(args.config)
    except Exception as e:
        print(f"Error running experiments from config: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()