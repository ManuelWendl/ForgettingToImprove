"""
Plotting utilities for Bayesian Optimization results.

This module provides functions to visualize optimization results,
including objective function contours, sampled points, and regret plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, Any
from tueplots import bundles, figsizes
from seaborn import axes_style


def _apply_iclr_style():
    """Apply ICLR 2023 styling to matplotlib."""
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=1))
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})
    # Set hatch line width to make hatching thinner
    plt.rcParams.update({"hatch.linewidth": 0.3})
    # Use normal legend font size
    plt.rcParams.update({"legend.fontsize": plt.rcParams['font.size']})


def _get_color_palette():
    """Get the standard color palette matching plot_gp.py."""
    return {
        'true_function': "#5F4690",    # Blue-purple
        'samples': "#0F8554",          # Green
        'prediction': "#CC503E",       # Red
        'uncertainty': "#CC503E",      # Red (same as prediction)
        'region': 'lightgreen',        # Light green for region
        'contour': 'viridis',          # Colormap for contours
        'scatter': 'plasma',           # Colormap for scatter plots
        'method_colors':  [
            "#5F4690",
            "#1D6996",
            "#38A6A5",
            "#0F8554",
            "#73AF48",
            "#EDAD08",
            "#E17C05",
            "#CC503E",
            "#94346E",
            "#6F4070",
            "#994E95",
            "#666666",
        ]
    }


def _style_axis(ax):
    """Apply consistent axis styling."""
    # Add grid with specific styling
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    # Add formatter for scientific notation only on y-axis
    y_formatter = ticker.ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(y_formatter)
    
    # Use plain formatter for x-axis (no exponential notation)
    x_formatter = ticker.ScalarFormatter(useOffset=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)


def plot_results(results: Dict[str, Any], config: Dict[str, Any], plot_path: str) -> None:
    """
    Create plots of the results.
    
    Args:
        results: Results dictionary with method names as keys and statistics as values
        config: Configuration dictionary
        plot_path: Directory path to save plots
    """
    # Apply ICLR 2023 styling
    _apply_iclr_style()
    
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    colors = _get_color_palette()
    
    # Get methods from results, with 'joint' (Forgetful BO) first
    methods = list(results.keys())
    if 'joint' in methods:
        methods.remove('joint')
        methods.insert(0, 'joint')
    
    # Get number of iterations from first method
    first_method = methods[0]
    n_iters = len(results[first_method]['simple_regret']['mean'])
    iterations = np.arange(1, n_iters + 1)
    
    # Calculate figure size for single plots
    single_figsize = (figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][0]/2, 
                      figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][1])
    
    # Plot simple regret
    fig, ax = plt.subplots(figsize=single_figsize)
    _style_axis(ax)
    
    all_means = []
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        # Mean line
        ax.semilogy(iterations, stats['simple_regret']['mean'], 
                   color=method_color, linewidth=2, label=f'{method.capitalize().replace('_', ' ') if method not in ["none", "joint"] else ("Homoscedastic GP" if method == "none" else "Forgetful BO")}')
        
        # Std bands with light fill + hatching
        mean_regret = stats['simple_regret']['mean']
        std_regret = stats['simple_regret']['std']
        all_means.extend(mean_regret)
        
        # Light background fill
        ax.fill_between(iterations, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret, 
                        facecolor=method_color, alpha=0.2, 
                        linewidth=0, edgecolor='none')
        # Hatching overlay
        ax.fill_between(iterations, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret, 
                        facecolor='none', 
                        linewidth=0, edgecolor=method_color)
    
    # Set y-axis limit based on mean values
    if all_means:
        mean_val = np.mean(all_means)
        if mean_val > 0:
            upper_limit = 10 ** (np.ceil(np.log10(mean_val)) + 1)
            ax.set_ylim(top=upper_limit)
    
    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel('Simple Regret')
    ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/simple_regret_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot cumulative regret
    fig, ax = plt.subplots(figsize=single_figsize)
    _style_axis(ax)
    
    all_means = []
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        mean_cumulative = stats['cumulative_regret']['mean']
        std_cumulative = stats['cumulative_regret']['std']
        all_means.extend(mean_cumulative)
        
        ax.plot(iterations, mean_cumulative, 
               color=method_color, linewidth=2, label=f'{method.capitalize().replace('_', ' ') if method not in ["none", "joint"] else ("Homoscedastic GP" if method == "none" else "Forgetful BO")}')
        
        # Std bands with light fill
        ax.fill_between(iterations,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        facecolor=method_color, alpha=0.2, 
                        linewidth=0, edgecolor='none')
        ax.fill_between(iterations,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        facecolor='none', 
                        linewidth=0, edgecolor=method_color)
    
    # Set y-axis limit based on mean values
    if all_means:
        mean_val = np.mean(all_means)
        if mean_val > 0:
            upper_limit = 10 ** (np.ceil(np.log10(mean_val)) + 1)
            ax.set_ylim(top=upper_limit)
    
    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/cumulative_regret_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot best values
    fig, ax = plt.subplots(figsize=single_figsize)
    _style_axis(ax)
    
    all_means = []
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        mean_best = stats['best_values']['mean']
        std_best = stats['best_values']['std']
        all_means.extend(mean_best)
        
        ax.plot(iterations, mean_best, 
               color=method_color, linewidth=2, label=f'{method.capitalize().replace('_', ' ') if method not in ["none", "joint"] else ("Homoscedastic GP" if method == "none" else "Forgetful BO")}')
        
        # Std bands with light fill
        ax.fill_between(iterations,
                        mean_best - std_best,
                        mean_best + std_best,
                        facecolor=method_color, alpha=0.2, 
                        linewidth=0, edgecolor='none')
        ax.fill_between(iterations,
                        mean_best - std_best,
                        mean_best + std_best,
                        facecolor='none', 
                        linewidth=0, edgecolor=method_color)
    
    # Set y-axis limit based on mean values
    if all_means:
        mean_val = np.mean([abs(v) for v in all_means])
        if mean_val > 0:
            max_abs = max([abs(v) for v in all_means])
            upper_limit = 10 ** (np.ceil(np.log10(max_abs)) + 1)
            lower_limit = -upper_limit if min(all_means) < 0 else None
            ax.set_ylim(top=upper_limit, bottom=lower_limit)
    
    ax.set_xlabel(r'Iteration $t$')
    ax.set_ylabel('Best Observed Value')
    ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/best_values_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_path}")


def plot_multiple_results(all_data: list, plot_path: str) -> None:
    """
    Create combined plots with multiple subplots for different experiments.
    
    Args:
        all_data: List of data dictionaries, each containing 'results' and 'config'
        plot_path: Directory path to save plots
    """
    # Apply ICLR 2023 styling
    _apply_iclr_style()
    
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    colors = _get_color_palette()
    
    n_experiments = len(all_data)
    
    # Determine subplot layout
    if n_experiments <= 4:
        n_rows = 1
        n_cols = n_experiments
    else:
        n_cols = 4
        n_rows = (n_experiments + n_cols - 1) // n_cols  # Ceiling division
    
    # Collect all methods across all experiments for shared legend
    # Put 'joint' (Forgetful BO) first
    all_methods = set()
    for data in all_data:
        all_methods.update(data['results'].keys())
    all_methods = sorted(list(all_methods))
    if 'joint' in all_methods:
        all_methods.remove('joint')
        all_methods.insert(0, 'joint')
    
    # Create method name mapping
    method_labels = {}
    for method in all_methods:
        if method in ["none", "joint"]:
            method_labels[method] = "Homoscedastic GP" if method == "none" else "Forgetful BO"
        else:
            method_labels[method] = method.capitalize().replace('_', ' ')
    
    # Calculate figure size
    fig_width = figsizes.iclr2023(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.1)['figure.figsize'][0]
    fig_height = figsizes.iclr2023(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.1)['figure.figsize'][1]
    
    # --- Plot 1: Simple Regret ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    # Flatten axes for easier indexing
    axes_flat = axes.flatten()
    
    # Create plots for each experiment
    for idx, (data, ax) in enumerate(zip(all_data, axes_flat[:n_experiments])):
        results = data['results']
        config = data['config']
        methods = list(results.keys())
        # Ensure 'joint' (Forgetful BO) is first
        if 'joint' in methods:
            methods.remove('joint')
            methods.insert(0, 'joint')
        
        # Get number of iterations
        first_method = methods[0]
        n_iters = len(results[first_method]['simple_regret']['mean'])
        iterations = np.arange(1, n_iters + 1)
        
        _style_axis(ax)
        
        all_means = []
        for i, method in enumerate(methods):
            stats = results[method]
            method_idx = all_methods.index(method)
            method_color = colors['method_colors'][method_idx % len(colors['method_colors'])]
            
            # Mean line
            ax.semilogy(iterations, stats['simple_regret']['mean'], 
                       color=method_color, linewidth=2, label=method_labels[method])
            
            # Std bands with light fill + hatching
            mean_regret = stats['simple_regret']['mean']
            std_regret = stats['simple_regret']['std']
            all_means.extend(mean_regret)
            
            # Light background fill
            ax.fill_between(iterations, 
                            mean_regret - std_regret, 
                            mean_regret + std_regret, 
                            facecolor=method_color, alpha=0.2, 
                            linewidth=0.5, edgecolor=method_color)
        
        # Set y-axis limit based on mean values
        if all_means:
            mean_val = np.min(all_means)
            if mean_val > 0:
                lower_limit = 10 ** (np.floor(np.log10(mean_val)))
                ax.set_ylim(bottom=lower_limit)
        
        # Only show x-label on last row of subplots
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'Iteration $t$')
        # Only show y-label on leftmost subplots
        if idx % n_cols == 0:
            ax.set_ylabel('Simple Regret')
        ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()}")
    
    # Hide unused subplots
    for idx in range(n_experiments, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Create shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(all_methods), 
           bbox_to_anchor=(0.5, 1.0), frameon=False,
           columnspacing=1.0,    # Reduce space between columns
           handlelength=1.5,     # Shorter legend lines
           handletextpad=0.5)    # Less space between line and text
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"{plot_path}/combined_simple_regret.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 2: Cumulative Regret ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (data, ax) in enumerate(zip(all_data, axes_flat[:n_experiments])):
        results = data['results']
        config = data['config']
        methods = list(results.keys())
        # Ensure 'joint' (Forgetful BO) is first
        if 'joint' in methods:
            methods.remove('joint')
            methods.insert(0, 'joint')
        
        first_method = methods[0]
        n_iters = len(results[first_method]['cumulative_regret']['mean'])
        iterations = np.arange(1, n_iters + 1)
        
        _style_axis(ax)
        
        all_means = []
        for i, method in enumerate(methods):
            stats = results[method]
            method_idx = all_methods.index(method)
            method_color = colors['method_colors'][method_idx % len(colors['method_colors'])]
            
            mean_cumulative = stats['cumulative_regret']['mean']
            std_cumulative = stats['cumulative_regret']['std']
            all_means.extend(mean_cumulative)
            
            ax.plot(iterations, mean_cumulative, 
                   color=method_color, linewidth=2, label=method_labels[method])
            
            # Std bands
            ax.fill_between(iterations,
                            mean_cumulative - std_cumulative,
                            mean_cumulative + std_cumulative,
                            facecolor=method_color, alpha=0.2, 
                            linewidth=0, edgecolor='none')
        
        # Only show x-label on last row of subplots
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'Iteration $t$')
        # Only show y-label on leftmost subplots
        if idx % n_cols == 0:
            ax.set_ylabel('Cumulative Regret')
        ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()}", pad=12)
    
    # Hide unused subplots
    for idx in range(n_experiments, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Create shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(all_methods), 
           bbox_to_anchor=(0.5, 0.95), frameon=False,
           columnspacing=1.0,    # Reduce space between columns
           handlelength=1.5,     # Shorter legend lines
           handletextpad=0.5)    # Less space between line and text
    
    plt.tight_layout()
    plt.savefig(f"{plot_path}/combined_cumulative_regret.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # --- Plot 3: Best Values ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = axes.flatten()
    
    for idx, (data, ax) in enumerate(zip(all_data, axes_flat[:n_experiments])):
        results = data['results']
        config = data['config']
        methods = list(results.keys())
        # Ensure 'joint' (Forgetful BO) is first
        if 'joint' in methods:
            methods.remove('joint')
            methods.insert(0, 'joint')
        
        first_method = methods[0]
        n_iters = len(results[first_method]['best_values']['mean'])
        iterations = np.arange(1, n_iters + 1)
        
        _style_axis(ax)
        
        all_means = []
        for i, method in enumerate(methods):
            stats = results[method]
            method_idx = all_methods.index(method)
            method_color = colors['method_colors'][method_idx % len(colors['method_colors'])]
            
            mean_best = stats['best_values']['mean']
            std_best = stats['best_values']['std']
            all_means.extend(mean_best)
            
            ax.plot(iterations, mean_best, 
                   color=method_color, linewidth=2, label=method_labels[method])
            
            # Std bands
            ax.fill_between(iterations,
                            mean_best - std_best,
                            mean_best + std_best,
                            facecolor=method_color, alpha=0.2, 
                            linewidth=0, edgecolor='none')
        
        # Only show x-label on last row of subplots
        if idx >= (n_rows - 1) * n_cols:
            ax.set_xlabel(r'Iteration $t$')
        # Only show y-label on leftmost subplots
        if idx % n_cols == 0:
            ax.set_ylabel('Best Observed Value')
        ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()}")
    
    # Hide unused subplots
    for idx in range(n_experiments, len(axes_flat)):
        axes_flat[idx].axis('off')
    
    # Create shared legend
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(all_methods), 
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(f"{plot_path}/combined_best_values.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined plots saved to {plot_path}")


def plot_runtime_statistics(results: Dict[str, Any], config: Dict[str, Any], plot_path: str) -> None:
    """
    Create bar plot of runtime statistics for each method.
    
    Args:
        results: Results dictionary with method names as keys and statistics as values
        config: Configuration dictionary
        plot_path: Directory path to save plots
    """
    # Apply ICLR 2023 styling
    _apply_iclr_style()
    
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    colors = _get_color_palette()
    
    # Get methods from results, with 'joint' (Forgetful BO) first
    methods = list(results.keys())
    if 'joint' in methods:
        methods.remove('joint')
        methods.insert(0, 'joint')
    
    # Check if timing data is available
    has_timing = all('timing' in results[method] for method in methods)
    if not has_timing:
        print("Warning: No timing data available in results")
        return
    
    # Extract timing statistics
    mean_times = []
    std_times = []
    min_times = []
    max_times = []
    method_labels = []
    
    for method in methods:
        timing = results[method]['timing']
        mean_times.append(timing['mean'])
        std_times.append(timing['std'])
        min_times.append(timing['min'])
        max_times.append(timing['max'])
        
        # Create nice label
        if method == 'none':
            method_labels.append('Homoscedastic\nGP')
        elif method == 'joint':
            method_labels.append('Forgetful BO')
        else:
            method_labels.append(method.capitalize().replace('_', '\n'))
    
    # Create figure with 2x2 subplots (mean±std, min, max, total time)
    fig, axes = plt.subplots(2, 2, figsize=(figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][0], 
                                             figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][1]))
    
    x_pos = np.arange(len(methods))
    bar_width = 0.6
    
    # Get colors for each method
    method_colors = [colors['method_colors'][i % len(colors['method_colors'])] for i in range(len(methods))]
    
    # Plot 1: Mean ± Std
    ax = axes[0, 0]
    bars = ax.bar(x_pos, mean_times, bar_width, yerr=std_times, 
                   color=method_colors, alpha=0.7, capsize=5, edgecolor='black', linewidth=1.25)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Mean ± Std (per iteration)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=8)
    ax.set_yscale('log')
    _style_axis(ax)
    ax.grid(axis='y')
    
    # Plot 2: Min
    ax = axes[0, 1]
    bars = ax.bar(x_pos, min_times, bar_width, 
                   color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.25)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Minimum (per iteration)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=8)
    ax.set_yscale('log')
    _style_axis(ax)
    ax.grid(axis='y')
    
    # Plot 3: Max
    ax = axes[1, 0]
    bars = ax.bar(x_pos, max_times, bar_width, 
                   color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.25)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Maximum (per iteration)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=8)
    ax.set_yscale('log')
    _style_axis(ax)
    ax.grid(axis='y')
    
    # Plot 4: Total time
    ax = axes[1, 1]
    total_times = [results[method]['timing']['total'] for method in methods]
    bars = ax.bar(x_pos, total_times, bar_width, 
                   color=method_colors, alpha=0.7, edgecolor='black', linewidth=1.25)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Total Time (all iterations)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=0, ha='center', fontsize=8)
    ax.set_yscale('log')
    _style_axis(ax)
    ax.grid(axis='y')
    
    # Create shared legend using patches
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=method_colors[i], edgecolor='black', 
                            linewidth=1.25, alpha=0.7, label=method_labels[i].replace('\n', ' '))
                      for i in range(len(methods))]
    fig.legend(handles=legend_elements, loc='lower center', ncol=len(methods),
              bbox_to_anchor=(0.5, -0.05), frameon=False,
              columnspacing=1.0, handlelength=1.5, handletextpad=0.5)
    
    plt.suptitle(f"Runtime Statistics: {config['objective'].replace('_', ' ').replace('botorch ', '').title()}", 
                 y=1.02, fontsize=12)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"{plot_path}/runtime_statistics_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Runtime statistics plot saved to {plot_path}")


def plot_multiple_runtime_statistics(all_data: list, plot_path: str) -> None:
    """
    Create combined runtime bar plots with subplots for different experiments.
    
    Args:
        all_data: List of data dictionaries, each containing 'results' and 'config'
        plot_path: Directory path to save plots
    """
    # Apply ICLR 2023 styling
    _apply_iclr_style()
    
    Path(plot_path).mkdir(parents=True, exist_ok=True)
    colors = _get_color_palette()
    
    n_experiments = len(all_data)
    
    # Determine subplot layout
    if n_experiments <= 4:
        n_rows = 1
        n_cols = n_experiments
    else:
        n_cols = 4
        n_rows = (n_experiments + n_cols - 1) // n_cols  # Ceiling division
    
    # Collect all methods across all experiments
    all_methods = set()
    for data in all_data:
        all_methods.update(data['results'].keys())
    all_methods = sorted(list(all_methods))
    if 'joint' in all_methods:
        all_methods.remove('joint')
        all_methods.insert(0, 'joint')
    
    # Create method name mapping
    method_labels = {}
    for method in all_methods:
        if method == 'none':
            method_labels[method] = 'Homoscedastic\nGP'
        elif method == 'joint':
            method_labels[method] = 'Forgetful BO'
        else:
            method_labels[method] = method.capitalize().replace('_', '\n')
    
    # Calculate figure size
    fig_width = figsizes.iclr2023(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.5)['figure.figsize'][0]
    fig_height = figsizes.iclr2023(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.5)['figure.figsize'][1]
    
    # Create 4 separate figures for each statistic type
    stat_types = [
        ('mean', 'Mean ± Std (per iteration)', True),
    ]
    
    for stat_type, stat_title, show_std in stat_types:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
        axes_flat = axes.flatten()
        
        for idx, (data, ax) in enumerate(zip(all_data, axes_flat[:n_experiments])):
            results = data['results']
            config = data['config']
            methods = list(results.keys())
            
            # Check if timing data is available
            has_timing = all('timing' in results[method] for method in methods)
            if not has_timing:
                ax.text(0.5, 0.5, 'No timing data', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            
            # Ensure 'joint' (Forgetful BO) is first
            if 'joint' in methods:
                methods.remove('joint')
                methods.insert(0, 'joint')
            
            # Extract timing statistics for this experiment
            if stat_type == 'total':
                stat_values = [results[method]['timing']['total'] for method in methods]
                yerr = None
            else:
                stat_values = [results[method]['timing'][stat_type] for method in methods]
                if show_std:
                    yerr = [results[method]['timing']['std'] for method in methods]
                else:
                    yerr = None
            
            x_pos = np.arange(len(methods))
            bar_width = 0.6
            
            # Get colors for each method
            method_colors = [colors['method_colors'][all_methods.index(m) % len(colors['method_colors'])] 
                            for m in methods]
            
            # Create bars
            bars = ax.bar(x_pos, stat_values, bar_width, yerr=yerr,
                          color=method_colors, alpha=0.7, capsize=3 if yerr is not None else 0, 
                          edgecolor='black', linewidth=1.0)
            
            # Only show y-label on leftmost subplots
            if idx % n_cols == 0:
                ax.set_ylabel('Time (seconds)')
            ax.set_yscale('log')
            
            ax.set_title(f"{config['objective'].replace('_', ' ').replace('botorch ', '').title()}", fontsize=9, pad=12)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([method_labels[m] for m in methods], rotation=0, ha='center', fontsize=7)
            _style_axis(ax)
            ax.grid(axis='y')
        
        # Hide unused subplots
        for idx in range(n_experiments, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        # Create shared legend using patches
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors['method_colors'][i % len(colors['method_colors'])], 
                                edgecolor='black', linewidth=1.0, alpha=0.7, 
                                label=method_labels[method].replace('\n', ' '))
                          for i, method in enumerate(all_methods)]
        fig.legend(handles=legend_elements, loc='lower center', ncol=len(all_methods),
                  bbox_to_anchor=(0.5, -0.02), frameon=False,
                  columnspacing=1.0, handlelength=1.5, handletextpad=0.5)
        
        plt.suptitle(f"Runtime Statistics: {stat_title}", y=0.98, fontsize=12)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.12)
        plt.savefig(f"{plot_path}/combined_runtime_{stat_type}.pdf", dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Combined runtime statistics plots saved to {plot_path}")
