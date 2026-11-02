"""Figures of the transductive benchmarks, made from the files in results/.

    # Bar comparison of all methods on the six benchmarks
    # (newest comparison_<benchmark>_method_noise_*.txt of each, from run_transductive.sh)
    python -m forgetting_to_improve.forgetting_to_improve.plot_transductive bar

    # Noise ablation, joint vs aleatoric-only removal criterion
    # (newest comparison_<benchmark>_noise_*_per_seed.json of the noise_ablation_* configs)
    python -m forgetting_to_improve.forgetting_to_improve.plot_transductive noise_ablation

Use --files to plot specific result files instead of the newest ones.
"""
import argparse
import glob
import json
import os

from .helper.plot_statistic_results import (
    parse_results_file,
    plot_bar_comparison,
    plot_noise_ablation_bars_grid,
    plot_noise_ablation_paired,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# objective -> panel name in the bar figure (panels are ordered by this name)
BAR_BENCHMARKS = {
    'twitter': 'twitter',
    'yacht': 'yacht',
    'boston': 'boston',
    'casp': 'casp',
    'sin_symmetric_lengthscale_increase': 'symmetric',
    'sin_asymmetric_lengthscale_increase': 'asymmetric',
}

# (panel title, objective) of the noise ablation, left to right
NOISE_ABLATION_BENCHMARKS = [
    ('Asymmetric', 'sin_asymmetric_lengthscale_increase'),
    ('Twitter', 'twitter'),
]


def _newest(pattern):
    """Newest file matching pattern (file names end in a sortable timestamp)."""
    files = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No results matching {pattern} in {RESULTS_DIR}")
    return files[-1]


def _table_objective(path):
    with open(path) as f:
        for line in f:
            if line.startswith('objective:'):
                return line.split(':', 1)[1].strip()
    raise ValueError(f"No 'objective:' line in {path}")


def bar(files, output_folder):
    if not files:
        files = [_newest(f"comparison_{objective}_method_noise_*.txt") for objective in BAR_BENCHMARKS]
    benchmarks = {}
    for path in files:
        print(f"Using {path}")
        benchmarks[BAR_BENCHMARKS[_table_objective(path)]] = parse_results_file(path)
    plot_bar_comparison(benchmarks, output_folder)


def noise_ablation(files, output_folder):
    if not files:
        files = [_newest(f"comparison_{objective}_noise_*_per_seed.json")
                 for _, objective in NOISE_ABLATION_BENCHMARKS]
    by_objective = {json.load(open(path))['objective']: path for path in files}
    benchmarks = [(title, by_objective[objective]) for title, objective in NOISE_ABLATION_BENCHMARKS
                  if objective in by_objective]
    for _, path in benchmarks:
        print(f"Using {path}")
    plot_noise_ablation_bars_grid(benchmarks, output_folder=output_folder)
    plot_noise_ablation_paired(benchmarks, output_folder=output_folder)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('figure', choices=['bar', 'noise_ablation'])
    parser.add_argument('--files', nargs='+', default=None,
                        help='Result files to plot (default: the newest of each benchmark in results/)')
    parser.add_argument('-o', '--output-folder', default=FIGURES_DIR)
    args = parser.parse_args()
    if args.figure == 'bar':
        bar(args.files, args.output_folder)
    else:
        noise_ablation(args.files, args.output_folder)


if __name__ == '__main__':
    main()
