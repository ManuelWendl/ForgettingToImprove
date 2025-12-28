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
        'method_colors': [             # Colors for different methods
            "#5F4690",  # Green
            "#CC503E",  # Dark orange  
            "#8B0000",  # Dark red
            "#4B0082",  # Indigo
            "#FF1493",  # Deep pink
            "#008B8B",  # Dark cyan
            "#9ACD32",  # Yellow green
            "#FF4500",  # Orange red
        ]
    }


def _style_axis(ax):
    """Apply consistent axis styling."""
    # Add grid with specific styling
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    # Add formatter for scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)


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
    
    # Get methods from results
    methods = list(results.keys())
    
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
    
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        # Mean line
        ax.semilogy(iterations, stats['simple_regret']['mean'], 
                   color=method_color, linewidth=2, label=f'{method.capitalize()}')
        
        # Std bands with light fill + hatching
        mean_regret = stats['simple_regret']['mean']
        std_regret = stats['simple_regret']['std']
        
        # Light background fill
        ax.fill_between(iterations, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret, 
                        facecolor=method_color, alpha=0.15, 
                        linewidth=0, edgecolor='none')
        # Hatching overlay
        ax.fill_between(iterations, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret, 
                        facecolor='none', hatch='//////', 
                        linewidth=0, edgecolor=method_color)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Simple Regret (log scale)')
    ax.set_title(f"{config['objective'].replace('_', ' ')} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/simple_regret_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot cumulative regret
    fig, ax = plt.subplots(figsize=single_figsize)
    _style_axis(ax)
    
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        mean_cumulative = stats['cumulative_regret']['mean']
        std_cumulative = stats['cumulative_regret']['std']
        
        ax.plot(iterations, mean_cumulative, 
               color=method_color, linewidth=2, label=f'{method.capitalize()}')
        
        # Std bands with light fill + hatching
        ax.fill_between(iterations,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        facecolor=method_color, alpha=0.15, 
                        linewidth=1, edgecolor='none')
        ax.fill_between(iterations,
                        mean_cumulative - std_cumulative,
                        mean_cumulative + std_cumulative,
                        facecolor='none', hatch='//////', 
                        linewidth=1, edgecolor=method_color)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cumulative Regret')
    ax.set_title(f"{config['objective'].replace('_', ' ')} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/cumulative_regret_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot best values
    fig, ax = plt.subplots(figsize=single_figsize)
    _style_axis(ax)
    
    for i, method in enumerate(methods):
        stats = results[method]
        method_color = colors['method_colors'][i % len(colors['method_colors'])]
        
        mean_best = stats['best_values']['mean']
        std_best = stats['best_values']['std']
        
        ax.plot(iterations, mean_best, 
               color=method_color, linewidth=2, label=f'{method.capitalize()}')
        
        # Std bands with light fill + hatching
        ax.fill_between(iterations,
                        mean_best - std_best,
                        mean_best + std_best,
                        facecolor=method_color, alpha=0.15, 
                        linewidth=0, edgecolor='none')
        ax.fill_between(iterations,
                        mean_best - std_best,
                        mean_best + std_best,
                        facecolor='none', hatch='//////', 
                        linewidth=0, edgecolor=method_color)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Best Observed Value')
    ax.set_title(f"{config['objective'].replace('_', ' ')} - {config['acquisition']}")
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(f"{plot_path}/best_values_{config['objective']}.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {plot_path}")
