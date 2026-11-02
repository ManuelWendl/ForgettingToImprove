"""Which real transitions Forgetful RL removes on cartpole balance.

Scatter of retained vs removed samples in (pole angle, angular velocity) at
three filtering events, next to the passive cartpole phase portrait with the
region where the dynamics are close to linear (the linear-kernel GP model is
accurate there).

Needs a forgetful run with overrides.store_data_removal=true (see
scripts/run_rl_cartpole.sh). Usage (from the repository root):
    python -m forgetting_to_improve.forgetful_rl.plotting.plot_removal_visualization \
        --runs-dir exp/mbpo/cartpole_balance_removal
"""
import argparse
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from seaborn import axes_style
from tueplots import bundles, figsizes

from .run_loader import get_cfg, load_local_runs

warnings.filterwarnings("ignore")

NUM_STEPS = 14000
SEED = 1
# Filtering events shown (0-based; one event per model update, every 200 steps)
EPISODES_IDX = [0, 44, 68]


def _is_removal_run(cfg) -> bool:
    return (
        get_cfg(cfg, "overrides.num_steps") == NUM_STEPS
        and bool(get_cfg(cfg, "overrides.store_data_removal", False))
        and bool(get_cfg(cfg, "overrides.forgetful_rl", False))
        and cfg.get("seed") == SEED
    )


def _style_axis(ax):
    """Apply consistent axis styling."""
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    y_formatter = ticker.ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(y_formatter)
    x_formatter = ticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)


def _wrap_theta(vals):
    # raw theta accumulates without wrapping (episodes can complete full
    # rotations since early termination is off): wrap into [-pi, pi]
    return np.mod(vals + np.pi, 2 * np.pi) - np.pi


def plot(removal_lists, output: str) -> None:
    # aggregate the selected filtering events across runs
    aggregated_tables = []
    for epi in EPISODES_IDX:
        dfs = [tables[epi] for tables in removal_lists
               if epi < len(tables) and tables[epi] is not None and len(tables[epi]) > 0]
        aggregated_tables.append(pd.concat(dfs, ignore_index=True) if dfs else None)
    if all(df is None for df in aggregated_tables):
        raise RuntimeError("No removal data found for the selected filtering events.")

    usetex = shutil.which("latex") is not None
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(bundles.neurips2024(usetex=usetex))
    plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=len(EPISODES_IDX), height_to_width_ratio=1.0))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})

    theta_lim = (-np.pi, np.pi)
    omega_lim = (-10, 10)
    i, j = 2, 3  # pole angle, pole angular velocity

    fig3, axes3 = plt.subplots(1, 4)
    for col in range(4):
        ax = axes3[col]
        _style_axis(ax)
        if col < 3:
            epi = EPISODES_IDX[col]
            df = aggregated_tables[col]
            if df is None:
                continue
            obs_arr = np.stack(df["obs"].to_list())
            removed = df["removed"].astype(bool).to_numpy()
            ax.scatter(_wrap_theta(obs_arr[~removed, i]), obs_arr[~removed, j], c="#1D6996", s=1, alpha=0.3)
            ax.scatter(_wrap_theta(obs_arr[removed, i]), obs_arr[removed, j], c="#CC503E", s=1, alpha=0.3)
            ax.set_title(f"Episode {epi + 1}")
            ax.set_xlabel("Pole Angle")
            if col == 0:
                ax.set_ylabel("Angular Velocity")
        else:
            # Phase portrait of the passive (F=0) cartpole pole dynamics, with
            # the same constants/equations as env/cartpole_continuous.py.
            gravity = 9.8
            masscart = 1.0
            masspole = 0.1
            total_mass = masscart + masspole
            length = 0.5  # half the pole's length
            polemass_length = masspole * length

            theta_vals = np.linspace(*theta_lim, 200)
            omega_vals = np.linspace(*omega_lim, 200)
            T, W = np.meshgrid(theta_vals, omega_vals)
            costheta = np.cos(T)
            sintheta = np.sin(T)
            temp = (polemass_length * W**2 * sintheta) / total_mass
            thetaacc = (gravity * sintheta - costheta * temp) / (
                length * (4.0 / 3.0 - masspole * costheta**2 / total_mass)
            )

            # Small-angle linearization about theta = theta_dot = 0; shade where
            # the nonlinear and linearized accelerations agree within 10%.
            l_eff = length * (4.0 / 3.0 - masspole / total_mass)
            thetaacc_lin = gravity * T / l_eff
            rel_err = np.abs(thetaacc - thetaacc_lin) / (np.abs(thetaacc_lin) + 1e-2)
            linear_ok = rel_err < 0.1
            ax.contourf(T, W, linear_ok.astype(float), levels=[0.5, 1.5],
                        colors=["#1D6996"], alpha=0.18, zorder=1)
            ax.contour(T, W, linear_ok.astype(float), levels=[0.5],
                       colors=["#1D6996"], linewidths=1.0, zorder=2)
            ax.streamplot(T, W, W, thetaacc, color="gray", linewidth=0.8, density=0.75, arrowsize=0.5, zorder=3)
            ax.add_patch(plt.Rectangle((-0.3, -0.3), 0.6, 0.6, fill=True, color="green", linewidth=0, zorder=4, alpha=0.7))
            ax.set_title("Phase Diagram")
            ax.set_xlabel("Pole Angle")

        ax.set_xlim(theta_lim)
        ax.set_ylim(omega_lim)
        ax.set_aspect((theta_lim[1] - theta_lim[0]) / (omega_lim[1] - omega_lim[0]), adjustable="box")
        ax.set_xticks([-np.pi, 0, np.pi])
        ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
        ax.set_yticks([-10, -5, 0, 5, 10])
        ax.set_yticklabels([r"-10", r"-5", "0", r"5", r"10"])

    linear_label = "Linear regime ($<$10\\% error)" if usetex else "Linear regime (<10% error)"
    handles = [
        Line2D([0], [0], marker="o", color="w", label="Retained samples", markerfacecolor="#1D6996", markersize=5),
        Line2D([0], [0], marker="o", color="w", label="Removed samples", markerfacecolor="#CC503E", markersize=5),
        Line2D([0], [0], color="gray", lw=1, label="CartPole Dynamics (F=0)"),
        Line2D([0], [0], color="green", lw=0, label="Goal", marker="s", markersize=5, alpha=0.7),
        Line2D([0], [0], color="#1D6996", lw=0, label=linear_label, marker="s", markersize=5, alpha=0.3),
    ]
    fig3.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.93), ncol=5, frameon=False,
                columnspacing=1.0, handletextpad=0.4)
    try:
        fig3.tight_layout()
    except Exception as e:  # noqa: BLE001
        print("warning: tight_layout failed", e)
    fig3.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", required=True, help="Folder with local Hydra run folders (searched recursively)")
    parser.add_argument("-o", "--output", default="removal_scatter.pdf", help="Output PDF")
    args = parser.parse_args()

    runs = [r for r in load_local_runs(args.runs_dir, where=_is_removal_run) if r["removal"]]
    print(f"Loaded {len(runs)} run(s) with removal data: " + ", ".join(
        f"{r['path']} ({len(r['removal'])} filtering events)" for r in runs))
    plot([r["removal"] for r in runs], args.output)


if __name__ == "__main__":
    main()
