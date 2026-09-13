"""
Script to load and plot saved learning histories without re-running experiments.

Usage:
    Single file:
    python -m forgetting_to_improve.forgetfull_bayesian_optimization.helper.plot_saved_results -f ./forgetting_to_improve/forgetfull_bayesian_optimization/results/botorch_ackley_2d_statistical.txt -o custom_figure_folder/

    Folder (all results in one plot):
    python -m forgetting_to_improve.forgetfull_bayesian_optimization.helper.plot_saved_results -f ./forgetting_to_improve/forgetfull_bayesian_optimization/results/stable_baseline/ -o custom_figure_folder/

    Multiple folders (combine results from different folders, per matching benchmark filename):
    python -m forgetting_to_improve.forgetfull_bayesian_optimization.helper.plot_saved_results -f ./results/stable_baseline/ ./results/stable_baseline_other/ -o custom_figure_folder/
"""

import argparse
from pathlib import Path
from .result_writer import load_learning_history
from .plotting import plot_results, plot_multiple_results, plot_runtime_statistics, plot_multiple_runtime_statistics


def combine_results_from_files(file_paths):
    """
    Load and combine results from multiple files (same experiment/benchmark, different baselines).

    Args:
        file_paths: List of paths to result files

    Returns:
        Combined data dictionary with all algorithms from all files
    """
    combined_results = {}
    combined_config = None

    for file_path in file_paths:
        data = load_learning_history(str(file_path))

        # Use config from first file
        if combined_config is None:
            combined_config = data['config']

        # Merge results dictionaries
        for algorithm, stats in data['results'].items():
            if algorithm in combined_results:
                print(f"  Warning: Algorithm '{algorithm}' appears in multiple files. Using from {file_path.name}")
            combined_results[algorithm] = stats

    return {'config': combined_config, 'results': combined_results}


def main():
    parser = argparse.ArgumentParser(description='Plot saved learning histories')
    parser.add_argument('-f', '--file', type=str, nargs='+', required=True,
                        help='Path(s) to saved results file or folder(s). Can specify multiple folders to combine results.')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output directory for plots (default: figures/)')

    args = parser.parse_args()

    input_paths = [Path(p) for p in args.file]

    # Determine output path
    if args.output:
        plot_path = args.output
    else:
        plot_path = 'forgetting_to_improve/forgetfull_bayesian_optimization/figures/'

    Path(plot_path).mkdir(parents=True, exist_ok=True)

    all_dirs = all(p.is_dir() for p in input_paths)

    if len(input_paths) > 1 and all_dirs:
        # Multiple folders mode: combine results from corresponding files, matched by filename
        print(f"Combining results from {len(input_paths)} folders:")
        for p in input_paths:
            print(f"  - {p}")

        result_files = sorted(list(input_paths[0].glob('*_statistical.txt')))
        if not result_files:
            print(f"No result files (*_statistical.txt) found in {input_paths[0]}")
            return

        print(f"\nFound {len(result_files)} result files to combine")

        all_data = []
        for result_file in result_files:
            result_filename = result_file.name
            print(f"\nProcessing {result_filename}...")

            matching_files = []
            for folder in input_paths:
                matching_file = folder / result_filename
                if matching_file.exists():
                    print(f"  Found in {folder.name}")
                    matching_files.append(matching_file)
                else:
                    print(f"  Not found in {folder.name}")

            if not matching_files:
                print(f"  Skipping {result_filename} - not found in any folder")
                continue

            combined_data = combine_results_from_files(matching_files)
            all_data.append(combined_data)

        if not all_data:
            print("\nNo data to plot!")
            return

        print(f"\nCreating combined plots in {plot_path}...")
        plot_multiple_results(all_data, plot_path)

        print(f"\nCreating combined runtime statistics plots in {plot_path}...")
        plot_multiple_runtime_statistics(all_data, plot_path)

        print("\nPlotting complete!")

    elif len(input_paths) == 1 and input_paths[0].is_dir():
        # Single folder mode (original behavior)
        input_path = input_paths[0]
        result_files = sorted(list(input_path.glob('*_statistical.txt')))
        if not result_files:
            print(f"No result files (*_statistical.txt) found in {input_path}")
            return

        print(f"Found {len(result_files)} result files in {input_path}")

        # Load all learning histories
        all_data = []
        for result_file in result_files:
            print(f"Loading {result_file.name}...")
            data = load_learning_history(str(result_file))
            all_data.append(data)

        # Create combined plots
        print(f"\nCreating combined plots in {plot_path}...")
        plot_multiple_results(all_data, plot_path)

        # Create combined runtime plots
        print(f"\nCreating combined runtime statistics plots in {plot_path}...")
        plot_multiple_runtime_statistics(all_data, plot_path)

        print("\nPlotting complete!")

    elif len(input_paths) == 1 and input_paths[0].is_file():
        # Single file mode (original behavior)
        input_file = str(input_paths[0])
        print(f"Loading learning history from {input_file}...")
        data = load_learning_history(input_file)

        results = data['results']
        config = data['config']
        # Create runtime plots
        print(f"\nCreating runtime statistics plot in {plot_path}...")
        plot_runtime_statistics(results, config, plot_path)


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

    else:
        print("Error: Mixed or unsupported input types.")
        print("Please provide either:")
        print("  - A single file")
        print("  - A single folder")
        print("  - Multiple folders (to combine)")
        return


if __name__ == '__main__':
    main()
