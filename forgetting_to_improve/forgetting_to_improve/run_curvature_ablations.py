"""Gamma noise-prior ablation of the restricted curvature (appendix figure).

Runs the three restricted-curvature Gamma-prior-a ablations back-to-back
(symmetric / asymmetric / twitter objectives) and combines their
lambda_min/lambda_max overlays into one figure with three subplots sharing a
single legend.

Each ablation is run exactly as `main.py -c <config>` would run it (same
results/figures side effects), via run_experiments_from_config(...,
collect_curvature_histories=True), which additionally returns the
gamma_prior_a-sweep curvature-history overlay for that config's objective.
The three overlays are then combined with plot_curvature_condition_grid.

Usage (from the repository root):
    python -m forgetting_to_improve.forgetting_to_improve.run_curvature_ablations

Output: figures/curvature_ablation_grid.pdf
"""
import os

from .main import run_experiments_from_config
from .helper.plot_gp import plot_curvature_condition_grid

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')

# (subplot title, config file) -- run left to right in this order.
ABLATIONS = [
    ("Symmetric", "ablation_gamma_prior_a_curvature_symmetric.yaml"),
    ("Asymmetric", "ablation_gamma_prior_a_curvature_asymmetric.yaml"),
    ("Twitter", "ablation_gamma_prior_a_curvature_twitter.yaml"),
]


def run_curvature_ablations():
    panels = []
    for panel_title, config_file in ABLATIONS:
        config_path = os.path.join(CONFIG_DIR, config_file)
        print("=" * 60)
        print(f"Running curvature ablation: {panel_title} ({config_file})")
        print("=" * 60)

        _, curvature_histories_by_objective = run_experiments_from_config(
            config_path, collect_curvature_histories=True
        )
        # Each ablation config sweeps gamma_prior_a for a single objective,
        # so exactly one entry is expected here.
        if not curvature_histories_by_objective:
            print(f"WARNING: no curvature history collected for {panel_title} ({config_file}); skipping panel")
            continue
        curvature_histories = next(iter(curvature_histories_by_objective.values()))
        panels.append((panel_title, curvature_histories))

    print("=" * 60)
    print("Building combined curvature-ablation figure")
    print("=" * 60)
    plot_curvature_condition_grid(panels, filename_stem="curvature_ablation_grid")


if __name__ == "__main__":
    run_curvature_ablations()
