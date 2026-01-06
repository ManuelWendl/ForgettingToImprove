"""
Script to load and plot saved learning histories without re-running experiments.

Usage:
    python -m forgetting_to_improve.forgetfull_bayesian_optimization.helper.plot_saved_results -f ./forgetting_to_improve/forgetfull_bayesian_optimization/results/botorch_ackley_2d_statistical.txt -o custom_figure_folder/
"""

import argparse
from pathlib import Path
from .result_writer import load_learning_history
from .plotting import plot_results


def main():
    parser = argparse.ArgumentParser(description='Plot saved learning histories')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='Path to saved results file (e.g., results/experiment_results.txt or results/experiment_results_history.pkl)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for plots (default: figures/)')
    
    args = parser.parse_args()
    
    # Load the learning history
    print(f"Loading learning history from {args.file}...")
    data = load_learning_history(args.file)
    
    results = data['results']
    config = data['config']
    
    # Determine output path
    if args.output:
        plot_path = args.output
    else:
        plot_path = 'forgetting_to_improve/forgetfull_bayesian_optimization/figures/'
    
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    
    # Create plots
    print(f"\nCreating plots in {plot_path}...")
    plot_results(results, config, plot_path)
    
    print("\nPlotting complete!")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for method_name, stats in results.items():
        print(f"\n{method_name.upper()}:")
        print(f"  Final Simple Regret: {stats['simple_regret']['mean'][-1]:.6f} ± {stats['simple_regret']['std'][-1]:.6f}")
        print(f"  Final Best Value: {stats['best_values']['mean'][-1]:.6f} ± {stats['best_values']['std'][-1]:.6f}")


if __name__ == '__main__':
    main()
