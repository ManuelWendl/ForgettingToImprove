"""Cartpole balance (linear-kernel GP-MBPO) learning curves: Forgetful RL vs GP-MBPO.

Three panels: evaluation return, number of removed real samples per
filtering step, and relative target-region size |A_n| / |S|. Mean over
seeds with standard-error bands.

Usage (from the repository root; scripts/run_rl_cartpole.sh writes the runs
to exp/mbpo/cartpole_balance/):
    python -m forgetting_to_improve.forgetful_rl.plotting.plot_cartpole_balance \
        --runs-dir exp/mbpo/cartpole_balance
"""
import argparse
import shutil
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn.objects as so
from matplotlib.lines import Line2D
from seaborn import axes_style
from tueplots import bundles, figsizes

from .run_loader import evaluation_frame, get_cfg, load_local_runs

warnings.filterwarnings("ignore")

NUM_STEPS = 14000


def _is_balance_run(cfg) -> bool:
    return (
        get_cfg(cfg, "overrides.num_steps") == NUM_STEPS
        and not get_cfg(cfg, "overrides.store_data_removal", False)
        and get_cfg(cfg, "dynamics_model.kernel_type") == "linear"
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


def plot(data: pd.DataFrame, output: str) -> None:
    # only keep seeds 0..4 and runs with an explicit forgetful_rl flag
    data = data[data["seed"].isin(range(5))]
    data = data[data["forgetful"].isin([False, True])]
    if data.empty:
        raise RuntimeError("No matching runs; check --runs-dir.")

    # truncate to the shortest run (smallest final env_step over seed/forgetful groups)
    max_steps = data.reset_index().groupby(["seed", "forgetful"])["env_step"].max()
    data = data[data.index <= max_steps.min()]

    metrics = [
        "eval/episode_reward",               # left, wide
        "forgetful_rl/removed_samples",      # center, quarter width
        "forgetful_rl/target_region_ratio",  # right, quarter width
    ]
    y_labels = {
        metrics[0]: r"$\hat R(\pi_n)$",
        metrics[1]: r"\# removed",
        metrics[2]: r"$|\mathcal{A}_n|/|\mathcal{S}|$",
    }
    usetex = shutil.which("latex") is not None
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(bundles.neurips2024(usetex=usetex))
    plt.rcParams.update(figsizes.neurips2024(nrows=1, ncols=3, height_to_width_ratio=0.7))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    if not usetex:
        y_labels[metrics[1]] = r"# removed"
    fig, axes = plt.subplots(1, 3, gridspec_kw={"width_ratios": [0.5, 0.25, 0.25]})

    marker_styles = ["", "", "", "", ""]
    colors = ["#5F4690", "#0F8554", "#73AF48", "#EDAD08", "#E17C05",
              "#CC503E", "#94346E", "#6F4070", "#994E95", "#666666"]
    text = {"False": r"GP-MBPO", "True": r"Forgetful RL"}

    for i, ax in enumerate(axes.flatten()):
        _style_axis(ax)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        so.Plot(
            data,
            x="env_step",
            marker="forgetful",
            color="forgetful",
            y=metrics[i],
        ).add(
            so.Line(linewidth=2.0, pointsize=2.5, edgewidth=0.1),
            so.Agg("mean"),
        ).add(so.Band(alpha=0.15), so.Est("mean", errorbar="se"), legend=False).scale(
            color=so.Nominal(values=colors, order=[True, False]),
            marker=so.Nominal(values=marker_styles, order=[True, False]),
        ).label(
            x="Steps",
            y=lambda name: y_labels[name],
        ).theme(axes_style("ticks")).on(ax).plot()

    axes = fig.get_axes()
    if fig.legends:
        fig.legends.pop(0)
    fig.legends = []

    for ax in axes:
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))  # forces scientific notation
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)

    # cap the "removed samples" subplot's y-axis
    axes[1].set_ylim(0, 1000)

    legend_handles = [
        Line2D([0], [0], color=colors[0], marker=marker_styles[0], linestyle="-", label=text["True"], markersize=4),
        Line2D([0], [0], color=colors[1], marker=marker_styles[1], linestyle="-", label=text["False"], markersize=4),
    ]
    fig.legend(
        legend_handles,
        [h.get_label() for h in legend_handles],
        loc="center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=2,
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.5,
        handlelength=1.5,
    )
    fig.savefig(output, bbox_inches="tight")
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", required=True, help="Folder with local Hydra run folders (searched recursively)")
    parser.add_argument("-o", "--output", default="cartpoleBalanceLinear.pdf", help="Output PDF")
    args = parser.parse_args()

    runs = list(load_local_runs(args.runs_dir, where=_is_balance_run))
    print(f"Loaded {len(runs)} runs: " + ", ".join(
        f"seed={r['seed']}/forgetful={r['forgetful']}" for r in runs))
    plot(pd.concat([evaluation_frame(r) for r in runs]), args.output)


if __name__ == "__main__":
    main()
