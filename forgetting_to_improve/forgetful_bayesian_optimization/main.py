import argparse
import os
from .helper.read_config import read_config, get_acquisition_config, get_optimization_config, get_perturbation_config, validate_config
from .run_experiment import run_experiment
from .helper.plotting import plot_results
from .helper.result_writer import save_results, save_learning_history, compute_statistics
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

    # handle special CSV-based AutoML benchmark
    if isinstance(config['objective'], str) and config['objective'].startswith('automl_csv:'):
        # path follows the colon
        csv_path = config['objective'].split(':', 1)[1]
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV benchmark file not found: {csv_path}")
        from .automl_benchmarks.csv_benchmark import (
            load_csv_benchmark, get_param_columns,
            compute_bounds, CsvObjective, SwitchingCsvObjective,
        )
        # read data once
        full_df = load_csv_benchmark(csv_path)

        # list of unique tasks in sorted order so experiments are reproducible
        tasks = sorted(full_df['data_id'].unique())
        # allow user to restrict to first N tasks for quicker/debug runs
        task_limit = config.get('task_limit', None)
        if task_limit is not None:
            tasks = tasks[:task_limit]
            print(f"Limiting CSV benchmark to first {task_limit} tasks (using {len(tasks)} actual)")
        n_tasks = len(tasks)
        n_iter_per_task = config['n_iter']
        total_iters = n_iter_per_task * n_tasks

        # build schedule and per-iteration optimal values
        # we require one extra entry corresponding to the initial sample
        schedule = []  # data_id for each step, including initialization
        optimal_values = []
        # start with the first task for the initial design; optimal values
        # should be positive `max accuracy` (note: CsvObjective stores
        # ``_optimal_value`` as -max_accuracy, which is why earlier code
        # negated it).  Here we are constructing the list ourselves so avoid
        # the extra negation.
        first_opt = float(full_df[full_df['data_id'] == tasks[0]]['accuracy'].max())
        schedule.append(tasks[0])
        optimal_values.append(first_opt)
        for t in tasks:
            df_t = full_df[full_df['data_id'] == t]
            opt = float(df_t['accuracy'].max())
            # each task contributes n_iter_per_task iterations
            schedule.extend([t] * n_iter_per_task)
            optimal_values.extend([opt] * n_iter_per_task)
        # verify lengths
        assert len(schedule) == total_iters + 1
        assert len(optimal_values) == total_iters + 1

        param_cols = get_param_columns(full_df)
        # bounds computed from full dataset to allow all parameters
        bounds = compute_bounds(full_df, param_cols)

        # create switching objective that can update its internal task
        obj_func = SwitchingCsvObjective(full_df, param_cols, schedule)

        # rest of experiment is identical to normal case, except we call
        # run_experiment once with ``total_iters`` and later use the
        # array of optimal values when computing statistics.
        acquisition_config = get_acquisition_config(config)
        acq_func = get_acquisition_function(acquisition_config)
        opti_options = get_optimization_config(config)
        perturbation_config = get_perturbation_config(config)

        algorithm = opti_options.get('algorithm', None)
        if not isinstance(algorithm, list):
            algorithm = [algorithm]

        results = {}
        for a in algorithm:
            opti_options['algorithm'] = a

            def initialize_model(train_x, train_y, bounds):
                return initialize_model_with_config(train_x, train_y, bounds, config)

            reset_per_task = config.get('sample_optimization', {}).get('reset_per_task', False)
            if reset_per_task:
                print(f"\nRunning continuous CSV benchmark ({total_iters} iters over {n_tasks} tasks), algorithm={a}...")
            # for CSV tasks we always use two initial points; later the
            # optimization loop will reset to two points when the task
            # changes as well.  This overrides any user-provided ``init_points``
            # setting in the configuration.
            best_observed_all, iteration_times_all = run_experiment(
                aq_func=acq_func,
                obj=obj_func,
                initialize_model=initialize_model,
                num_restarts=10,
                raw_samples=512,
                obs_noise=0.0,
                n_initial_samples=2,
                global_bounds=bounds,
                perturbation_config=perturbation_config if perturbation_config else None,
                num_iters=total_iters,
                num_seeds=config['n_seeds'],
                opti_options=opti_options if opti_options else None,
            )

            print("\nComputing statistics...")
            stats = compute_statistics(best_observed_all, optimal_values, iteration_times_all, task_schedule=schedule)
            results[a] = stats

            # Print per-task summary if available
            if 'per_task' in stats:
                print("\nPer-task regret summary:")
                for task_id, task_stats in stats['per_task'].items():
                    print(f"  Task {task_id}: Final regret = {task_stats['simple_regret']['final_mean']:.6f} ± {task_stats['simple_regret']['final_std']:.6f}")

            # save/plot as in normal path
            if config.get('save_results', False):
                results_path = config.get('results_path', 'experiment_results.txt')
                out_path = f"forgetting_to_improve/forgetfull_bayesian_optimization/results/{results_path}"
                print(f"\nSaving results to {out_path}...")
                save_results(results, config, out_path)
                save_learning_history(results, config, out_path)
            if config.get('plot_results', False):
                plot_path = 'forgetting_to_improve/forgetfull_bayesian_optimization/figures/'
                print("\nCreating plots for continuous CSV experiment...")
                plot_results(results, config, plot_path)

            print(f"\nFinal simple regret (mean) after {total_iters} iterations: {stats['simple_regret']['mean'][-1]:.6f}")

        # we handled CSV case; exit
        return
    
    # Get objective function and bounds
    obj_func, bounds = get_objective_function(config['objective'])
    optimal_value = -obj_func._optimal_value  # Negate because we negate the function
    
    # Get acquisition function
    acquisition_config = get_acquisition_config(config)
    acq_func = get_acquisition_function(acquisition_config)
    
    # Get optimization configuration
    opti_options = get_optimization_config(config)
    
    # Get perturbation configuration
    perturbation_config = get_perturbation_config(config)

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
        best_observed_all, iteration_times_all = run_experiment(
            aq_func=acq_func,
            obj=obj_func,
            initialize_model=initialize_model,
            num_restarts=10,
            raw_samples=512,
            obs_noise=0.0,  # Noise-free observations by default
            n_initial_samples=config.get('init_points', 0) if config.get('init_points', 0) > 0 else 5,
            global_bounds=bounds,
            perturbation_config=perturbation_config if perturbation_config else None,
            num_iters=config['n_iter'],
            num_seeds=config['n_seeds'],
            opti_options=opti_options if opti_options else None,
        )

        # Compute statistics
        print("\nComputing statistics...")
        stats = compute_statistics(best_observed_all, optimal_value, iteration_times_all)
        
        results[a] = stats

    # Save results if requested
    if config.get('save_results', False):
        results_path = config.get('results_path', 'experiment_results.txt')
        results_path = 'forgetting_to_improve/forgetfull_bayesian_optimization/results/' + results_path
        print(f"\nSaving results to {results_path}...")
        save_results(results, config, results_path)
        # Save complete learning history for plotting
        save_learning_history(results, config, results_path)
    
    # Plot results if requested
    if config.get('plot_results', False):
        plot_path = 'forgetting_to_improve/forgetfull_bayesian_optimization/figures/'
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
