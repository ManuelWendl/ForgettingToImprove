"""Figures of the transductive benchmarks: the bar comparison of all methods
(plot_bar_comparison) and the joint vs aleatoric-only noise ablation
(plot_noise_ablation_bars_grid, plot_noise_ablation_paired). See
plot_transductive.py for the command line."""
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
from seaborn import axes_style
from tueplots import bundles, figsizes

from .plot_gp import latex_available


def parse_results_file(filepath):
    """
    Parse a results .txt file and extract method statistics.
    
    Parameters
    ----------
    filepath : str
        Path to the results file
        
    Returns
    -------
    dict
        Dictionary mapping method names to their metrics
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract the results section
    results_section = re.search(r'=== Results ===(.*?)(?:=== Minimum Values|$)', content, re.DOTALL)
    if not results_section:
        return None
    
    results_text = results_section.group(1)
    
    # Parse each line
    results = {}
    for line in results_text.strip().split('\n'):
        if line.startswith('---') or line.startswith('Variant'):
            continue
        if not line.strip():
            continue
            
        # Split by whitespace, handling multiple spaces
        parts = line.split()
        if len(parts) < 2:
            continue
            
        method_name = parts[0]
        
        # Extract mean values (format: "value ± std")
        metrics = {}
        metric_names = ['CE', 'I-ECE', 'MAE', 'MSE', 'NLL', 'RMSE', 'Runtime', 'Sharpness', 'V-ECE']
        
        # Find positions of ± symbols
        plus_minus_positions = [i for i, p in enumerate(parts) if '±' in p]
        
        # Extract metrics based on expected order
        metric_idx = 0
        for i in range(1, len(parts)):
            if '±' in parts[i]:
                # parts[i-1] is the mean, parts[i+1] is the std
                if i > 0 and metric_idx < len(metric_names):
                    try:
                        mean_val = float(parts[i-1])
                        metrics[metric_names[metric_idx]] = mean_val
                        metric_idx += 1
                    except (ValueError, IndexError):
                        pass
        
        if metrics:
            results[method_name] = metrics
    
    return results


def normalize_metrics(data, metrics, methods_to_include=None):
    """
    Normalize metrics to [0, 1] range for radar plot.
    Lower values are better for all metrics.
    Worst (max) value is mapped to 0, best (min) value is mapped to 1.
    
    Parameters
    ----------
    data : dict
        Dictionary mapping method names to their metrics
    metrics : list
        List of metric names to normalize
    methods_to_include : list, optional
        List of method keys to include in normalization. If None, uses all methods.
        
    Returns
    -------
    dict
        Normalized data
    """
    normalized = {}
    
    # Filter data to only include specified methods
    if methods_to_include is not None:
        filtered_data = {k: v for k, v in data.items() if k in methods_to_include}
    else:
        filtered_data = data
    
    # Find min and max for each metric across filtered methods only
    min_vals = {}
    max_vals = {}
    for metric in metrics:
        values = [filtered_data[method][metric] for method in filtered_data if metric in filtered_data[method]]
        if values:
            min_vals[metric] = min(values)
            max_vals[metric] = max(values)
    
    # Normalize each method's metrics
    for method in data:
        normalized[method] = {}
        for metric in metrics:
            if metric in data[method]:
                val = data[method][metric]
                min_val = min_vals.get(metric, val)
                max_val = max_vals.get(metric, val)
                
                # Normalize so that worst (max) = 0 and best (min) = 1
                if max_val - min_val > 1e-10:
                    norm_val = (max_val - val) / (max_val - min_val)
                else:
                    norm_val = 0.5
                
                normalized[method][metric] = norm_val
            else:
                normalized[method][metric] = 0.0
    
    return normalized


def plot_bar_comparison(benchmarks, output_folder=None):
    """
    Bar plot comparing all methods on each benchmark: one panel per benchmark,
    one group of bars per metric, scaled per metric so that the best method
    is 1 and the worst 0.

    Parameters
    ----------
    benchmarks : dict
        Benchmark name -> parsed results table (parse_results_file). Panels
        are ordered by name.
    output_folder : str, optional
        Where to save benchmark_bar_comparison.pdf (default: the package's figures/).
    """
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": latex_available()})
    plt.rcParams.update({"legend.frameon": False})
    
    # Method mapping (sequential_joint first to plot first)
    method_mapping = {
        'sequential_joint': 'Forgetting to Improve',
        'none_joint': 'Homoscedastic GP',
        'targetSampling_joint': 'Target Sampling',
        'relevance_pursuit_joint': 'Relevance Pursuit',
        'warped_gp_joint': 'Warped GP',
        'heteroscedastic_gp_joint': 'Heteroscedastic GP'
    }
    
    # Colors for each method (matching plot_gp.py style)
    method_colors = {
        'sequential_joint': '#5F4690',  # Purple for Forgetting to Improve
        'none_joint': "#1D6996",
        'targetSampling_joint': "#38A6A5",
        'relevance_pursuit_joint': "#0F8554",
        'warped_gp_joint': "#73AF48",
        'heteroscedastic_gp_joint': "#EDAD08"
    }
    
    # Metrics to plot
    metrics = ['MSE', 'MAE', 'NLL', 'CE', 'V-ECE', 'Sharpness']
    
    # Benchmark display name mapping
    benchmark_display_names = {
        'twitter': 'Dow Jones',
        'boston': 'Boston Housing',
        'casp': 'CASP',
        'yacht': 'Yacht'
    }
    
    # Determine grid size
    n_benchmarks = len(benchmarks)
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, 
                             figsize=figsizes.neurips2024(nrows=n_rows, ncols=n_cols, 
                                                          height_to_width_ratio=0.75)['figure.figsize'])
    
    # Ensure axes is always a list
    if n_benchmarks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_benchmarks > 1 else [axes]
    
    # Bar width and positions
    n_methods = len(method_mapping)
    bar_width = 0.8 / n_methods  # Total width of 0.8 divided by number of methods
    
    # Plot each benchmark
    for idx, (benchmark_name, data) in enumerate(sorted(benchmarks.items())):
        ax = axes[idx]
        
        # Normalize metrics (only considering methods we're plotting)
        normalized_data = normalize_metrics(data, metrics, methods_to_include=list(method_mapping.keys()))
        
        # X-axis positions for metrics
        x_positions = np.arange(len(metrics))
        
        # Plot bars for each method
        for method_idx, method_key in enumerate(method_mapping.keys()):
            if method_key in normalized_data:
                # Calculate offset for this method
                offset = (method_idx - n_methods/2 + 0.5) * bar_width
                values = [normalized_data[method_key].get(m, 0.0) for m in metrics]
                
                # Plot bars
                ax.bar(x_positions + offset, values,
                      width=bar_width,
                      color=method_colors[method_key],
                      label=method_mapping[method_key],
                      alpha=0.9 if method_key == 'sequential_joint' else 0.4,
                      edgecolor=method_colors[method_key],
                      linewidth=1.0)
        
        # Determine row and column position
        row = idx // n_cols
        col = idx % n_cols
        
        # Configure axes
        ax.set_xticks(x_positions)
        # Only show x-tick labels in the bottom row
        if row == n_rows - 1 or idx >= n_benchmarks - n_cols:
            ax.set_xticklabels(metrics, rotation=45, ha='right')
        else:
            ax.set_xticklabels([])
        
        # Only show y-label in the first column
        if col == 0:
            ax.set_ylabel('Rel. Performance', fontsize=9)
        
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        
        # Only show y-tick labels in the first column
        if col != 0:
            ax.set_yticklabels([])
        
        ax.grid(True, linewidth=0.5, color='gainsboro', alpha=0.5, axis='y')
        ax.set_axisbelow(True)
        
        # Set title with mapped display name
        display_name = benchmark_display_names.get(benchmark_name, benchmark_name.replace('_', ' ').title())
        ax.set_title(display_name, weight='bold', size=10, pad=10)
        
        # Add horizontal line at 1.0 for reference (best possible)
        ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.3, zorder=0)
    
    # Hide unused subplots
    for idx in range(n_benchmarks, len(axes)):
        axes[idx].axis('off')
    
    # Create shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        # Reorder so Forgetting to Improve appears first in legend
        try:
            fti_idx = labels.index('Forgetting to Improve')
            # Move it to the front
            handles = [handles[fti_idx]] + handles[:fti_idx] + handles[fti_idx+1:]
            labels = [labels[fti_idx]] + labels[:fti_idx] + labels[fti_idx+1:]
        except ValueError:
            pass  # If not found, keep original order
        
        fig.legend(handles, labels, 
                  loc='upper center',
                  bbox_to_anchor=(0.5, 0.1),
                  ncol=3,
                  frameon=False,
                  fontsize=9,
                  columnspacing=1.0,
                  handlelength=2.0)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    
    # Save figure
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, 'benchmark_bar_comparison.pdf')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Bar plot saved to {output_path}")
    plt.close()


def _latex_p(p):
    """p-value as LaTeX math, e.g. 'p = 1.1 \\cdot 10^{-5}'."""
    if not np.isfinite(p):
        return 'p = \\mathrm{n/a}'
    if p >= 1e-3:
        return f'p = {p:.2g}'
    mant, exp = f'{p:.1e}'.split('e')
    return f'p = {mant} \\cdot 10^{{{int(exp)}}}'


def plot_noise_ablation_paired(benchmarks, metrics=('RMSE', 'NLL'),
                               noise_a='joint', noise_b='aleatoric', output_folder=None,
                               filename='noise_ablation_paired.pdf'):
    """Per-seed paired scatter of two removal criteria (x: noise_a, y: noise_b).

    One panel per (metric, benchmark); one dot per seed. Seeds where both
    criteria give the identical model sit on the diagonal in gray; seeds that
    differ are colored. Points above y = x are seeds where noise_a is better
    (all metrics here are lower-is-better). Each panel reports the win counts
    and the paired Wilcoxon p-value over the differing seeds.

    Args:
        benchmarks: List of (panel title, per-seed results file) pairs; the
            files are the *_per_seed.json files written next to the tables.
        metrics: Metrics to plot, one row each.
        noise_a, noise_b: The two criteria to compare (x and y axis).
        output_folder: Where to save (defaults to the package figures dir).
        filename: Output file name.
    """
    from scipy.stats import wilcoxon

    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": latex_available()})
    plt.rcParams.update({"legend.frameon": False})

    color_diff = "#1D6996"
    color_same = "#B0B0B0"
    labels = {'joint': 'joint', 'aleatoric': 'aleatoric only'}

    n_rows, n_cols = len(metrics), len(benchmarks)
    figsize = figsizes.neurips2024(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.0)['figure.figsize']
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for col, (title, path) in enumerate(benchmarks):
        variants = json.load(open(path))['variants']
        ra, rb = variants[noise_a], variants[noise_b]
        seeds = sorted(set(ra) & set(rb), key=int)
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            a = np.array([ra[s][metric] for s in seeds])
            b = np.array([rb[s][metric] for s in seeds])
            same = np.isclose(a, b)

            ax.scatter(a[same], b[same], s=10, color=color_same, linewidth=0, zorder=2)
            ax.scatter(a[~same], b[~same], s=12, color=color_diff, linewidth=0, zorder=3)

            lo = min(a.min(), b.min())
            hi = max(a.max(), b.max())
            pad = 0.05 * (hi - lo)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color='black',
                    linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)
            log_axes = hi - lo > 10
            if log_axes:
                # Heavy right tail (a few badly overconfident seeds).
                ax.set_xscale('symlog', linthresh=1.0)
                ax.set_yscale('symlog', linthresh=1.0)
            else:
                ax.set_xlim(lo - pad, hi + pad)
                ax.set_ylim(lo - pad, hi + pad)
            if not log_axes:
                ax.set_aspect('equal', adjustable='box')

            d = b[~same] - a[~same]
            wins_a, wins_b = int((d > 0).sum()), int((d < 0).sum())
            p = wilcoxon(d).pvalue if len(d) > 5 else float('nan')
            ax.text(0.04, 0.96,
                    f"{labels[noise_a]} better: {wins_a}\n{labels[noise_b]} better: {wins_b}\n"
                    f"identical: {int(same.sum())}\n${_latex_p(p)}$",
                    transform=ax.transAxes, va='top', ha='left', fontsize=6)

            ax.grid(True, alpha=0.3, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
            if row == 0:
                ax.set_title(title)
            ax.set_xlabel(f"{metric} ({labels[noise_a]})")
            ax.set_ylabel(f"{metric} ({labels[noise_b]})")

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Noise ablation plot saved to {filepath}")
    plt.close(fig)


def _noise_ablation_panels(path, metrics, noise_a, noise_b):
    """Per-seed values for each panel: [(key, label, values_a, values_b)]."""
    data = json.load(open(path))
    num_train_samples = data['num_train_samples']
    ra, rb = data['variants'][noise_a], data['variants'][noise_b]
    seeds = sorted(set(ra) & set(rb), key=int)
    panels = []
    for m in metrics:
        panels.append((m, m, np.array([ra[s][m] for s in seeds]), np.array([rb[s][m] for s in seeds])))
    panels.append(('removed', r'\# removed' if plt.rcParams['text.usetex'] else '# removed',
                   np.array([num_train_samples - ra[s]['n_opt'] for s in seeds], dtype=float),
                   np.array([num_train_samples - rb[s]['n_opt'] for s in seeds], dtype=float)))
    return panels


def _apply_noise_ablation_style(rel_width):
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(bundles.neurips2024(rel_width=rel_width))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": latex_available()})


def plot_noise_ablation_bars_grid(benchmarks, metrics=('RMSE', 'NLL'),
                                  noise_a='joint', noise_b='aleatoric', output_folder=None,
                                  filename='noise_ablation_bars.pdf'):
    """Joint vs aleatoric-only removal criterion, one titled subfigure per
    benchmark side by side in one full-text-width figure. Each subfigure has
    one small panel per quantity (metrics and number of removed samples, each
    on its own y-scale, named by its x-label) with one bar per criterion
    (mean over seeds), styled like plot_bar_comparison
    (Forgetting-to-Improve purple for joint) with one shared legend.

    Args:
        benchmarks: List of (title, per-seed results file), left to right.
        metrics: Metrics to plot (lower is better).
        noise_a, noise_b: The two criteria (variant names in the per-seed file).
        output_folder: Where to save (defaults to the package figures dir).
        filename: Output file name.
    """
    from matplotlib.patches import Patch

    _apply_noise_ablation_style(1.0)
    colors = {noise_a: '#5F4690', noise_b: '#38A6A5'}
    alphas = {noise_a: 0.9, noise_b: 0.4}
    names = {noise_a: 'joint', noise_b: 'aleatoric only'}

    n_panels = len(metrics) + 1
    figsize = figsizes.neurips2024(nrows=1, ncols=n_panels * len(benchmarks),
                                   height_to_width_ratio=1.4)['figure.figsize']
    fig = plt.figure(figsize=figsize)
    subfigs = np.atleast_1d(fig.subfigures(1, len(benchmarks), wspace=0.12))
    for subfig, (title, path) in zip(subfigs, benchmarks):
        panels = _noise_ablation_panels(path, metrics, noise_a, noise_b)
        axes = np.atleast_1d(subfig.subplots(1, n_panels))
        for ax, (_, label, va, vb) in zip(axes, panels):
            means = [va.mean(), vb.mean()]
            for i, (noise, m) in enumerate(zip((noise_a, noise_b), means)):
                ax.bar(i, m, width=0.8, color=colors[noise], alpha=alphas[noise],
                       edgecolor=colors[noise], linewidth=1.0, zorder=2)
            ax.axhline(0, color='black', linewidth=0.6)
            lo, hi = min(0.0, *means), max(0.0, *means)
            span = hi - lo
            ax.set_ylim(lo - (0.1 * span if lo < 0 else 0.0), hi + 0.1 * span)
            ax.set_xticks([])
            ax.set_xlabel(label)
            ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
        subfig.suptitle(title)

    handles = [Patch(facecolor=colors[n], alpha=alphas[n], edgecolor=colors[n], linewidth=1.0)
               for n in (noise_a, noise_b)]
    fig.legend(handles, [names[noise_a], names[noise_b]], loc='upper center',
               bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Noise ablation bar plot saved to {filepath}")
    plt.close(fig)
