import argparse

from .helper.read_config import read_config, get_acquisition_config, get_optimization_config, validate_config
from .run_experiment import run_experiment
from .helper.plotting import plot_results
from .helper.result_writer import save_results, compute_statistics
from .helper.setup_experiment import get_objective_function, get_acquisition_function, initialize_model_with_config

def run_bayesian_optimization_experiment(config_path: str) -> None:
    """
    Main function to run Bayesian optimization experiments from config file.
    
    Args:
        config_path: Path to the YAML configuration file
    """
    # Read and validate configuration
    config = read_config(config_path)
    validate_config(config)
    
    print("=" * 80)
    print("Starting Bayesian Optimization Experiment")
    print("=" * 80)
    print(f"Objective: {config['objective']}")
    print(f"Acquisition: {config['acquisition']}")
    print(f"Number of iterations: {config['n_iter']}")
    print(f"Number of seeds: {config['n_seeds']}")
    print(f"Initial points: {config.get('init_points', 0)}")
    print("=" * 80)
    
    # Get objective function and bounds
    obj_func, bounds = get_objective_function(config['objective'])
    optimal_value = -obj_func._optimal_value  # Negate because we negate the function
    
    # Get acquisition function
    acquisition_config = get_acquisition_config(config)
    acq_func = get_acquisition_function(acquisition_config)
    
    # Get optimization configuration
    opti_options = get_optimization_config(config)

    algorithm = opti_options.get('algorithm', None)
    if not isinstance(algorithm, list):
        algorithm = [algorithm]

    results = {}

    for a in algorithm:
        opti_options['algorithm'] = a
    
        # Define initialize_model function that uses config
        def initialize_model(train_x, train_y, bounds):
            return initialize_model_with_config(train_x, train_y, bounds, config)
        
        # Run experiments
        print("\nRunning experiments...")
        best_observed_all = run_experiment(
            aq_func=acq_func,
            obj=obj_func,
            initialize_model=initialize_model,
            num_restarts=10,
            raw_samples=512,
            obs_noise=0.0,  # Noise-free observations by default
            n_initial_samples=config.get('init_points', 0) if config.get('init_points', 0) > 0 else 5,
            global_bounds=bounds,
            num_iters=config['n_iter'],
            num_seeds=config['n_seeds'],
            opti_options=opti_options if opti_options else None,
        )

        # Compute statistics
        print("\nComputing statistics...")
        stats = compute_statistics(best_observed_all, optimal_value)
        
        results[a] = stats

    # Save results if requested
    if config.get('save_results', False):
        results_path = config.get('results_path', 'experiment_results.txt')
        results_path = 'forgetfull_bayesian_optimization/results/' + results_path
        print(f"\nSaving results to {results_path}...")
        save_results(results, config, results_path)
    
    # Plot results if requested
    if config.get('plot_results', False):
        plot_path = 'forgetfull_bayesian_optimization/figures/'
        print("\nCreating plots...")
        plot_results(results, config, plot_path)
    
    # Print final statistics
    print("\n" + "=" * 80)
    print("Experiment Complete - Final Results")
    print("=" * 80)
    print(f"Final Simple Regret (mean ± std): {stats['simple_regret']['mean'][-1]:.6f} ± {stats['simple_regret']['std'][-1]:.6f}")
    print(f"Final Cumulative Regret (mean ± std): {stats['cumulative_regret']['mean'][-1]:.6f} ± {stats['cumulative_regret']['std'][-1]:.6f}")
    print(f"Final Best Value (mean ± std): {stats['best_values']['mean'][-1]:.6f} ± {stats['best_values']['std'][-1]:.6f}")
    print("=" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Run Bayesian Optimization experiments from config file')
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to the YAML configuration file')
    
    args = parser.parse_args()
    
    run_bayesian_optimization_experiment(args.config)


if __name__ == '__main__':
    main()
