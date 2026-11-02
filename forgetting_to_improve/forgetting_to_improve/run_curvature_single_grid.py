"""Run the three single-experiment restricted-curvature configs (symmetric,
asymmetric, twitter) and combine their curvature plots (lambda_min/lambda_max
of H_R and rho_k) into one figure with a shared legend -- see
plot_curvature_bounds_grid.

Usage (from the repository root):
    python -m forgetting_to_improve.forgetting_to_improve.run_curvature_single_grid

Output: figures/restricted_curvature_grid.pdf
"""

import os

from .main import run_single_experiment
from .helper.config_loader import ConfigLoader
from .helper.results_writer import ResultsWriter
from .helper.plot_gp import plot_curvature_bounds_grid


CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')

# (subplot title, config file) -- left to right in this order.
EXPERIMENTS = [
    ("Symmetric", "single_experiment_symmetric.yaml"),
    ("Asymmetric", "single_experiment_asymmetric.yaml"),
    ("Twitter", "single_experiment_twitter.yaml"),
]


def run_curvature_single_grid():
    panels = []
    for panel_title, config_file in EXPERIMENTS:
        config = ConfigLoader().load_config(os.path.join(CONFIG_DIR, config_file))
        print("=" * 60)
        print(f"Running restricted-curvature experiment: {panel_title} ({config_file})")
        print("=" * 60)
        _, _, plot_data = run_single_experiment(
            config, config['methods'][0], config['objectives'][0], config['objective_funcs'][0],
            config['objective_types'][0], config['noises'][0], config['seeds'][0], ResultsWriter(),
            return_plot_data=True,
        )
        panels.append((panel_title, (plot_data or {}).get('curvature_history')))

    plot_curvature_bounds_grid(panels, filename_stem="restricted_curvature_grid")


if __name__ == "__main__":
    run_curvature_single_grid()
