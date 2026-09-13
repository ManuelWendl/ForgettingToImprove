import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
from tueplots import bundles, figsizes
from seaborn import axes_style
import os
import numpy as np
from scipy import stats

def _uses_jacobian_kernel(kernel_config) -> bool:
    """True if a kernel_config (list of {'type': ..., ...} dicts, as parsed
    from a comparison_*_jacobian.yaml's `kernel:` block) includes the
    linearised-Laplace / Jacobian kernel."""
    if not kernel_config:
        return False
    if isinstance(kernel_config, dict):
        kernel_config = [kernel_config]
    return any(isinstance(k, dict) and str(k.get('type', '')).lower() == 'jacobian'
               for k in kernel_config)


def plot_predictions_comparison(objective_function, x, comparison_data, x_limits, A_limits, title=None, annotation=False, usetex=False):
    """
    Plot multiple GP predictions side-by-side with shared legend using ICLR 2023 style.

    Args:
        objective_function: The true objective function
        x: Test points
        comparison_data: Dict mapping method names to dicts with keys:
                        'x_samples', 'y_samples', 'y_pred', 'y_std'
        x_limits: Limits for x-axis
        A_limits: Limits for the region box
        title: Base title for the plot
        annotation: Whether to annotate sample indices
        usetex: If True, render text with a real LaTeX install instead of
            matplotlib's mathtext (used for the jacobian/NN-kernel
            comparison plots); defaults to False elsewhere to avoid
            requiring a LaTeX install.
    """
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": usetex})
    plt.rcParams.update({"legend.frameon": False})
    
    n_methods = len(comparison_data)
    method_names = list(comparison_data.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_methods, 
                            figsize=(figsizes.iclr2023(nrows=1, ncols=n_methods+1,height_to_width_ratio=1.5)['figure.figsize'][0], 
                                    figsizes.iclr2023(nrows=1, ncols=n_methods+1,height_to_width_ratio=1.5)['figure.figsize'][1]))
    
    # Handle single subplot case
    if n_methods == 1:
        axes = [axes]
    
    # Define colors matching the reference style
    colors = {
        'true_function': "#666666",    # Black
        'samples': "#5F4690",          # Blue-purple
        'prediction': "#1D6996",       # Blue
        'uncertainty': "#38A6A5",      # Teal
        'region': "#73AF48"         # Light green for region
    }
    
    # Create method title mapping
    method_titles = {
        'none': 'Homoscedastic GP',
        'heteroscedastic_gp': 'Heteroscedastic GP',
        'sequential': 'Forgetting to Improve',
        'batch': 'Batch Optimization',
        'targetSampling': 'Target Sampling',
        'relevance_pursuit': 'Relevance Pursuit',
        'warped_gp': 'Warped GP'
    }
    
    handles, labels = None, None
    
    for idx, (ax, method) in enumerate(zip(axes, method_names)):
        data = comparison_data[method]
        x_samples = data['x_samples']
        y_samples = data['y_samples']
        y_pred = data['y_pred']
        y_std = data['y_std']
        
        # Add formatter for scientific notation
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.xaxis.set_major_formatter(formatter)
        
        # Plot region box
        box = Rectangle((A_limits[0], min(objective_function(x))-3), 
                       A_limits[1] - A_limits[0], 
                       max(objective_function(x))+3 - (min(objective_function(x))-3), 
                       linewidth=1.25, edgecolor=colors['region'], 
                       facecolor=colors['region'], alpha=0.2, zorder=0, label=r'$\mathcal{A}$')
        ax.add_patch(box)
        
        # Plot true function
        ax.plot(x, objective_function(x), color=colors['true_function'], 
                linewidth=1.5, label=r'$f^*$', alpha=0.8)
        
        # Plot samples with specific marker style
        ax.scatter(x_samples, y_samples, c=colors['samples'], marker='o', 
                  s=25, label='Samples', edgecolors=colors['samples'], linewidth=0.1, zorder=5)
        
        # Add sample index annotations if requested
        if annotation:
            for i, (x_sample, y_sample) in enumerate(zip(x_samples, y_samples)):
                ax.annotate(str(i), (x_sample, y_sample), 
                           xytext=(3, 3), textcoords='offset points',
                           fontsize=6, color='black', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                    edgecolor='none', alpha=0.7))
        
        # Plot GP prediction
        ax.plot(x, y_pred, color=colors['prediction'], linestyle='-', 
                linewidth=1.5, label=r'$\mu_{\mathrm{GP}}(x)$')
        
        # Plot uncertainty band
        ax.fill_between(x, y_pred - 2.567 * y_std, y_pred + 2.567 * y_std, 
                        color=colors['uncertainty'], alpha=0.3, 
                        label=r'$\sigma_{\mathrm{GP}}(x)$')
        
        # Set labels with LaTeX formatting (only leftmost gets y-label)
        ax.set_xlabel('$x$')
        if idx == 0:
            ax.set_ylabel('$f(x)$')
        
        # Set y-limits and x-limits
        ax.set_ylim(min(objective_function(x))-1.5, max(objective_function(x))+1.5)
        ax.set_xlim(x_limits)
        
        # Set subplot title
        subplot_title = method_titles.get(method, method)
        ax.set_title(subplot_title, fontsize=9)
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
        
        # Collect handles and labels from the first subplot for shared legend
        if idx == 0:
            handles, labels = ax.get_legend_handles_labels()
    
    # Remove individual legends from subplots
    for ax in axes:
        if ax.get_legend():
            ax.get_legend().remove()
    
    # Create shared legend at the top
    if handles and labels:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(1.05, 0.5), 
                  ncol=1, frameon=False, columnspacing=0.8,
                  handletextpad=0.4, handlelength=1.2, fontsize=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    if title:
        filename = title.replace(" ", "_").lower() + '.pdf'
    else:
        filename = 'comparison_' + '_'.join(method_names).lower() + '.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Comparison plot saved to {filepath}")
    plt.close()

def plot_predictions(objective_function, x, x_samples, y_samples, y_pred, y_std, x_limits, A_limits, title=None, annotation=False):
    """
    Plot the GP predictions along with the true function using ICLR 2023 style.
    """
    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=1))  # Changed to ncols=2 for side-by-side
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})

    halfsided = False

    if halfsided:
        # Create figure with ICLR styling - specify figsize for half width
        fig, ax = plt.subplots(1, 1, figsize=(figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][0]/2, 
                                            figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][1]))
    else:
        # Create figure with proper ICLR styling
        fig, ax = plt.subplots(1, 1, figsize=(figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][0], 
                                            figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][1]))
        
    # Add formatter for scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Add grid with specific styling
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    
    # Define colors matching the reference style
    colors = {
        'true_function': "#5F4690",    # Blue-purple
        'samples': "#0F8554",          # Green
        'prediction': "#CC503E",       # Red
        'uncertainty': "#CC503E",      # Red (same as prediction)
        'region': 'lightgreen'         # Light green for region
    }
    
    # Plot region box
    box = Rectangle((A_limits[0], min(objective_function(x))-3), A_limits[1] - A_limits[0], max(objective_function(x))+3 - (min(objective_function(x))-3), linewidth=1.25, edgecolor='gray', 
                    facecolor=colors['region'], alpha=0.3, zorder=0, label=r'$\mathcal{A}$')
    ax.add_patch(box)
    
    # Plot true function
    ax.plot(x, objective_function(x), color=colors['true_function'], 
            linewidth=1.5, label=r'$f^*$', alpha=0.8)
    
    # Plot samples with specific marker style
    ax.scatter(x_samples, y_samples, c=colors['samples'], marker='o', 
              s=20, label='Samples', edgecolors=colors['samples'], linewidth=0.1, zorder=5)
    
    # Add sample index annotations if requested
    if annotation:
        for i, (x_sample, y_sample) in enumerate(zip(x_samples, y_samples)):
            ax.annotate(str(i), (x_sample, y_sample), 
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=6, color='black', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
    
    # Plot GP prediction
    ax.plot(x, y_pred, color=colors['prediction'], linestyle='--', 
            linewidth=1.5, label=r'$\mu_{\mathrm{GP}}(x)$')
    
    # Plot uncertainty band
    ax.fill_between(x, y_pred - 2.567 * y_std, y_pred + 2.567 * y_std, 
                    color=colors['uncertainty'], alpha=0.2, 
                    label=r'$\pm 2.567 \sigma_{\mathrm{GP}}(x)$')
    
    # Set labels with LaTeX formatting
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    
    # # Set y-limits
    ax.set_ylim(min(objective_function(x))-3, max(objective_function(x))+3)
    ax.set_xlim(x_limits)

    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    # Create legend with ICLR style - adjusted position for smaller figure
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.45), 
                      ncol=3, frameon=False, columnspacing=0.8,  # Reduced ncol and spacing
                      handletextpad=0.4, handlelength=1.2, fontsize=8)  # Smaller legend
    
    # Save the plot
    if title:
        filename = title.replace(" ", "_").lower() + '.pdf'
    else:
        filename = 'gp_predictions.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def clean_kernel_name(kernel_name):
    """Remove 'kernel_*_' prefix from kernel names for cleaner display."""
    import re
    # Remove pattern like 'kernel_0_', 'kernel_1_', etc.
    cleaned = re.sub(r'^kernel_\d+_', '', kernel_name)
    return cleaned


def plot_per_kernel_predictions(objective_function, x, x_samples, y_samples, y_pred, y_std, 
                               x_limits, A_limits, kernel_indicators=None, kernel_names=None, 
                               title=None, annotation=False):
    """
    Plot GP predictions with different colors for samples used by different kernels.
    
    Args:
        objective_function: True function to plot
        x: Test points
        x_samples: Training sample locations
        y_samples: Training sample values
        y_pred: GP predictions
        y_std: GP prediction standard deviations
        x_limits: X-axis limits
        A_limits: Active learning region limits
        kernel_indicators: Dict mapping kernel names to boolean arrays indicating which samples are used
        kernel_names: List of kernel names
        title: Plot title
        annotation: Whether to annotate sample indices
    """
    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][0],
                                          figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][1]))
    
    # Add formatter for scientific notation
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.xaxis.set_major_formatter(formatter)
    
    # Add grid
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    
    # Define colors
    colors = {
        'true_function': "#5F4690",    # Blue-purple
        'prediction': "#CC503E",       # Red
        'uncertainty': "#CC503E",      # Red (same as prediction)
        'region': 'lightgreen',        # Light green for region
        'unused': "#CCCCCC",           # Gray for unused samples
    }
    
    # Define distinct colors for different kernels
    kernel_colors = [
        "#0F8554",  # Green
        "#FF8C00",  # Dark orange  
        "#8B0000",  # Dark red
        "#4B0082",  # Indigo
        "#FF1493",  # Deep pink
        "#008B8B",  # Dark cyan
        "#9ACD32",  # Yellow green
        "#FF4500",  # Orange red
    ]
    
    # Plot region box
    box = Rectangle((A_limits[0], min(objective_function(x))-1), A_limits[1] - A_limits[0], max(objective_function(x))+1 - (min(objective_function(x))-1), linewidth=1.25, edgecolor='gray', 
                    facecolor=colors['region'], alpha=0.3, zorder=0, label=r'$\mathcal{A}$')
    ax.add_patch(box)
    
    # Plot true function
    ax.plot(x, objective_function(x), color=colors['true_function'], 
            linewidth=1.5, label=r'$f^*$', alpha=0.8)
    
    # Plot samples with different colors based on kernel usage
    if kernel_indicators is not None and kernel_names is not None:
        # Find which samples are used by which kernels
        sample_kernel_usage = {}
        for i in range(len(x_samples)):
            used_by_kernels = []
            for kernel_name in kernel_names:
                if kernel_name in kernel_indicators and kernel_indicators[kernel_name][i]:
                    used_by_kernels.append(kernel_name)
            sample_kernel_usage[i] = used_by_kernels
        
        # Group samples by their exact kernel usage patterns
        usage_patterns = {}
        for i in range(len(x_samples)):
            used_kernels = sample_kernel_usage[i]
            if len(used_kernels) == 0:
                pattern_key = "unused"
            elif len(used_kernels) == 1:
                pattern_key = clean_kernel_name(used_kernels[0])
            else:
                # Sort kernel names for consistent labeling
                pattern_key = "+".join(sorted([clean_kernel_name(k) for k in used_kernels]))
            
            if pattern_key not in usage_patterns:
                usage_patterns[pattern_key] = []
            usage_patterns[pattern_key].append(i)
        
        # Plot samples grouped by usage pattern
        plotted_samples = set()
        
        # First, plot samples used by individual kernels
        for j, kernel_name in enumerate(kernel_names):
            cleaned_kernel_name = clean_kernel_name(kernel_name)
            if cleaned_kernel_name in usage_patterns:
                indices = usage_patterns[cleaned_kernel_name]
                if indices:
                    color = kernel_colors[j % len(kernel_colors)]
                    ax.scatter(x_samples[indices], y_samples[indices], 
                              c=color, marker='o', s=20, 
                              label=f'{cleaned_kernel_name} only', 
                              edgecolors=color, linewidth=0.1, zorder=5)
                    plotted_samples.update(indices)
        
        # Plot samples used by multiple kernels with specific combination labels
        shared_pattern_colors = {}
        color_idx = len(kernel_names)  # Start after individual kernel colors
        
        cleaned_kernel_names = [clean_kernel_name(k) for k in kernel_names]
        for pattern_key, indices in usage_patterns.items():
            if pattern_key not in cleaned_kernel_names and pattern_key != "unused" and indices:
                # This is a shared pattern - assign it a unique color and marker
                if pattern_key not in shared_pattern_colors:
                    shared_pattern_colors[pattern_key] = kernel_colors[color_idx % len(kernel_colors)]
                    color_idx += 1
                
                color = shared_pattern_colors[pattern_key]
                ax.scatter(x_samples[indices], y_samples[indices], 
                          c=color, marker='s', s=25,  # Square marker for shared
                          label=pattern_key,
                          edgecolors=color, linewidth=0.5, zorder=6)
                plotted_samples.update(indices)
        
        # Plot unused samples
        if "unused" in usage_patterns and usage_patterns["unused"]:
            indices = usage_patterns["unused"]
            ax.scatter(x_samples[indices], y_samples[indices], 
                      c=colors['unused'], marker='x', s=20,
                      label='Unused',
                      edgecolors=colors['unused'], linewidth=1.0, zorder=4)
            plotted_samples.update(indices)
        
    else:
        # Fallback to regular plotting if no kernel information
        ax.scatter(x_samples, y_samples, c="#0F8554", marker='o', 
                  s=20, label='Samples', edgecolors="#0F8554", linewidth=0.1, zorder=5)
    
    # Add sample index annotations if requested
    if annotation:
        for i, (x_sample, y_sample) in enumerate(zip(x_samples, y_samples)):
            ax.annotate(str(i), (x_sample, y_sample), 
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=6, color='black', 
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                edgecolor='none', alpha=0.7))
    
    # Plot GP prediction
    ax.plot(x, y_pred, color=colors['prediction'], linestyle='--', 
            linewidth=1.5, label=r'$\mu_{\mathrm{GP}}(x)$')
    
    # Plot uncertainty band
    ax.fill_between(x, y_pred - 2.567 * y_std, y_pred + 2.567 * y_std, 
                    color=colors['uncertainty'], alpha=0.2, 
                    label=r'$\pm 2.567 \sigma_{\mathrm{GP}}(x)$')
    
    # Set labels with LaTeX formatting
    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    
    # Set limits
    ax.set_ylim(min(objective_function(x)), max(objective_function(x)))
    ax.set_xlim(x_limits)

    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    # Create legend - adjust based on number of items
    handles, labels = ax.get_legend_handles_labels()
    n_items = len(labels)
    
    if n_items <= 4:
        ncol = 2
        bbox_y = 1.35
        fontsize = 8
    elif n_items <= 8:
        ncol = 3
        bbox_y = 1.50
        fontsize = 6
    else:
        ncol = 4  
        bbox_y = 1.60
        fontsize = 6
    
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, bbox_y), 
              ncol=ncol, frameon=False, columnspacing=0.8,
              handletextpad=0.4, handlelength=1.2, fontsize=fontsize)
    
    # Save the plot
    if title:
        filename = title.replace(" ", "_").lower() + '.pdf'
    else:
        filename = 'per_kernel_gp_predictions.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()


def visualize_hessian(H, v, u, removed_point, title="Hessian Matrix"):
    """
    Visualize the Hessian matrix and vectors using heatmaps with ICLR 2023 style.
    """
    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=1))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})
    
    # Create figure with proper ICLR styling for 3 subplots with equal heights
    fig, axes = plt.subplots(1, 3, width_ratios=[20,1,1], constrained_layout=True,
                            figsize=(figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][0], 
                                    figsizes.iclr2023(nrows=1, ncols=1)['figure.figsize'][1]))

    ax1, ax2, ax3 = axes

    # Disable grid for all subplots (consistent heatmap style)
    for ax in axes:
        ax.grid(False)
    
    # Plot Hessian matrix
    cax1 = ax1.matshow(H, cmap='PRGn')
    
    # Mark the removed point's row and column in red on Hessian
    if removed_point is not None:
        # Mark the row
        ax1.axhline(y=removed_point, color="#CC503E", linewidth=2, alpha=0.8)
        # Mark the column
        ax1.axvline(x=removed_point, color="#CC503E", linewidth=2, alpha=0.8)
    
    # Add colorbar for Hessian
    cbar1 = fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(labelsize=10)
    
    # Set labels for Hessian
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Index')
    ax1.set_title(r'Hessian $\sigma_W$')
    
    # Plot v vector as heatmap (reshape to column vector)
    v_reshaped = v.reshape(-1, 1)  # Make it a column vector for display
    cax2 = ax2.matshow(v_reshaped, cmap='PRGn', aspect='auto')
    
    # Mark the removed point in red
    if removed_point is not None and removed_point < len(v):
        ax2.axhline(y=removed_point, color="#CC503E", linewidth=2, alpha=0.8)
    
    # Add colorbar for v
    cbar2 = fig.colorbar(cax2, ax=ax2, fraction=0.46, pad=0.3)
    cbar2.ax.tick_params(labelsize=10)
    
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Index')
    ax2.set_title(r'$v(x_t)$')
    ax2.set_xticks([])  # Remove x-axis ticks since it's just one column

    # Plot u vector as heatmap (reshape to column vector)
    u_reshaped = u.reshape(-1, 1)  # Make it a column vector for display
    cax3 = ax3.matshow(u_reshaped, cmap='PRGn', aspect='auto')

    print(u)
    # Mark the removed point in red
    if removed_point is not None and removed_point < len(u):
        ax3.axhline(y=removed_point, color="#CC503E", linewidth=2, alpha=0.8)
    
    # Add colorbar for u
    cbar3 = fig.colorbar(cax3, ax=ax3, fraction=0.46, pad=0.3)
    cbar3.ax.tick_params(labelsize=10)
    
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Index')
    ax3.set_title(r'$u(x_t)$')
    ax3.set_xticks([])  # Remove x-axis ticks since it's just one column
    
    # Style the spines for all subplots
    for ax in [ax1, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
    
    # Save the plot
    filename = title.replace(" ", "_").lower() + '.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()

def plot_calibration_curve(p_levels, p_hat, title="Calibration Curve"):
    """
    Plot the calibration curve using ICLR 2023 style.
    """
    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=3))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})
    
    # Create figure with proper ICLR styling
    fig, ax = plt.subplots(1, 1, figsize=(figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][0]/2, 
                                          figsizes.iclr2023(nrows=1, ncols=2)['figure.figsize'][1]))

    # Add grid
    ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
    
    # Plot calibration curve
    ax.bar(p_levels-(1/(2*len(p_levels))), p_hat, width=0.1, alpha=0.7, color='#5F4690', label='Empirical', zorder=4)

    # Plot ideal calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ideal', zorder=5)

    # Set labels with LaTeX formatting
    ax.set_xlabel('$p$')
    ax.set_ylabel('$\\hat{p}$')
    
    # Set limits and aspect
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal', adjustable='box')
    
    # Style the spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    # Save the plot
    filename = title.replace(" ", "_").lower() + '.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()


def plot_calibration_comparison(objective_function, x, comparison_data, title=None, A_limits=None, usetex=False):
    """
    Plot calibration curves for multiple methods side-by-side using ICLR 2023 style.

    Args:
        objective_function: The true objective function
        x: Test points
        comparison_data: Dict mapping method names to dicts with keys:
                        'x_samples', 'y_samples', 'y_pred', 'y_std'
        title: Base title for the plot
        A_limits: Target region limits (tuple for 1D or list of tuples for multidimensional)
                  If provided, calibration is computed only within this region
        usetex: If True, render text with a real LaTeX install instead of
            matplotlib's mathtext (used for the jacobian/NN-kernel
            comparison plots); defaults to False elsewhere to avoid
            requiring a LaTeX install.
    """
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": usetex})
    plt.rcParams.update({"legend.frameon": False})
    
    n_methods = len(comparison_data)
    method_names = list(comparison_data.keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_methods, 
                            figsize=(figsizes.iclr2023(nrows=1, ncols=n_methods, height_to_width_ratio=1.0)['figure.figsize'][0]/2, 
                                    figsizes.iclr2023(nrows=1, ncols=n_methods, height_to_width_ratio=1.0)['figure.figsize'][1]))
    
    # Handle single subplot case
    if n_methods == 1:
        axes = [axes]
    
    # Create method title mapping
    method_titles = {
        'none': 'Homoscedastic GP',
        'heteroscedastic_gp': 'Heteroscedastic GP',
        'sequential': 'Forgetting to Improve',
        'batch': 'Batch Optimization',
        'targetSampling': 'Target Sampling',
        'relevance_pursuit': 'Relevance Pursuit',
        'warped_gp': 'Warped GP'
    }
    
    p_levels = np.linspace(0.0, 1.0, 10)
    
    for idx, (ax, method) in enumerate(zip(axes, method_names)):
        data = comparison_data[method]
        y_pred = data['y_pred']
        y_std = data['y_std']
        
        # Get true values
        y_true = objective_function(x)
        
        # Filter data to target region if A_limits is provided
        if A_limits is not None:
            # Handle both 1D and multidimensional regions
            if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
                # 1D case
                x_flat = x.flatten() if x.ndim == 2 else x
                if isinstance(A_limits[0], (list, tuple)):
                    # Multidimensional region specified but data is 1D - use first dimension
                    min_x, max_x = A_limits[0]
                else:
                    # Simple 1D region
                    min_x, max_x = A_limits
                mask = (x_flat >= min_x) & (x_flat <= max_x)
            else:
                # Multidimensional case
                if not isinstance(A_limits[0], (list, tuple)):
                    # Same region for all dimensions
                    A_limits_expanded = [A_limits] * x.shape[1]
                else:
                    A_limits_expanded = A_limits
                
                mask = np.ones(len(x), dtype=bool)
                for i, (min_val, max_val) in enumerate(A_limits_expanded):
                    mask &= (x[:, i] >= min_val) & (x[:, i] <= max_val)
            
            # Apply mask to filter data to target region
            y_true = y_true[mask]
            y_pred = y_pred[mask]
            y_std = y_std[mask]
        
        # Compute calibration curve
        z = np.abs((y_true - y_pred) / y_std)  # standardized absolute residuals
        
        # Empirical coverage for each nominal central interval
        p_hat = np.empty(len(p_levels))
        for i, p in enumerate(p_levels):
            alpha = (1 - p) / 2.0
            z_thresh = stats.norm.ppf(1 - alpha)  # two-sided z threshold
            p_hat[i] = np.mean(z <= z_thresh)     # fraction inside central interval
        
        # Compute calibration error
        weights = np.ones(len(p_levels)) / len(p_levels)
        sq_diffs = (p_levels - p_hat) ** 2
        ce = float(np.sum(weights * sq_diffs))
        
        # Add grid
        ax.grid(True, linewidth=0.5, c="gainsboro", zorder=0)
        
        # Plot calibration bars
        ax.bar(p_levels-(1/(2*len(p_levels))), p_hat, width=0.1, alpha=0.7, 
               color='#5F4690', label='Empirical', zorder=4)
        
        # Plot ideal calibration line
        ax.plot([0, 1], [0, 1], linestyle='--', color='black', label='Ideal', zorder=5)
        
        # Set labels with LaTeX formatting (only leftmost gets y-label)
        ax.set_xlabel('$p$')
        if idx == 0:
            ax.set_ylabel('$\\hat{p}$')
        
        # Set limits and aspect
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        
        # Set subplot title with CE value
        subplot_title = method_titles.get(method, method)
        ax.set_title(f"{subplot_title}\nCE = {ce:.4f}", fontsize=9)
        
        # Style the spines
        for spine in ax.spines.values():
            spine.set_linewidth(1.25)
    
    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save the plot
    if title:
        filename = title.replace(" ", "_").lower() + '_calibration.pdf'
    else:
        filename = 'calibration_comparison_' + '_'.join(method_names).lower() + '.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Calibration comparison plot saved to {filepath}")
    plt.close()


def visualize_per_kernel_influence(epistemic_influences, title="Per-Kernel Influences"):
    """
    Visualize per-kernel epistemic and aleatoric influences using bar plots with ICLR 2023 style.
    """
    import numpy as np
    # Validate that all elements are numbers before converting
    # Ensure inputs are numpy float arrays
    epistemic_influences = np.array(epistemic_influences, dtype=float)

    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})      

    n_test_points = 10

    # Plot this as a colorplot with datapoinsts x kernels matrix and the influence values as colors
    fig, axes = plt.subplots(n_test_points, 1, figsize=(figsizes.iclr2023(nrows=n_test_points, ncols=1)['figure.figsize'][0], 
                                          figsizes.iclr2023(nrows=n_test_points, ncols=1)['figure.figsize'][1]))   
    testpoint = 0
    for ax in axes:
        ax.grid(False)  
        # Plot epistemic influences
        epistemic_influence = epistemic_influences[:, testpoint, :]
        cax1 = ax.matshow(epistemic_influence, cmap='viridis')
        cbar1 = fig.colorbar(cax1, ax=ax, fraction=0.046, pad=0.04, orientation='horizontal', location='bottom')
        cbar1.ax.tick_params(labelsize=10)
        ax.set_xlabel('Kernels')
        ax.set_ylabel('Data Points')
        ax.set_title('Epistemic Influences')

        for spine in ax.spines.values():
            spine.set_linewidth(1.25)

        testpoint += 1
    # Save the plot
    filename = title.replace(" ", "_").lower() + '.pdf'
    
    # Create figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close()


def plot_optimal_comparison(fti_uncertainty, optimal_uncertainty, n_samples_list, title="FTI vs Optimal Subset Selection"):
    """
    Plot comparison between FTI (greedy) and optimal (exhaustive search) subset selection.
    
    Args:
        fti_uncertainty: List of uncertainties from FTI greedy approach
        optimal_uncertainty: List of uncertainties from optimal exhaustive search
        n_samples_list: List of subset sizes corresponding to each uncertainty value
        title: Plot title
    """
    if not fti_uncertainty or not optimal_uncertainty:
        print("No uncertainty data to plot")
        return
    
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})
    
    # Create single plot
    fig, ax = plt.subplots(1, 1, 
                          figsize=(figsizes.iclr2023(nrows=1, ncols=1, height_to_width_ratio=0.6)['figure.figsize']))
    
    # Define colors
    color_fti = "#5F4690"  # Purple - Forgetting to Improve
    color_optimal = "#1D6996"  # Blue - Optimal
    
    # Convert to numpy arrays
    n_samples_array = np.array(n_samples_list)
    fti_array = np.array(fti_uncertainty)
    optimal_array = np.array(optimal_uncertainty)
    
    # Plot both curves
    ax.plot(n_samples_array, fti_array, marker='o', markersize=5, 
            color=color_fti, linewidth=2, alpha=0.9, label='FTI (Greedy)')
    ax.plot(n_samples_array, optimal_array, marker='s', markersize=5, 
            color=color_optimal, linewidth=2, alpha=0.9, label='Optimal (Exhaustive)')
    
    # Set labels
    ax.set_xlabel(r'Number of Samples $|\mathcal{S}|$')
    ax.set_ylabel(r'Predictive Uncertainty $\bar{\sigma}^2$')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add legend
    ax.legend(loc='best', frameon=False, fontsize=9)
    
    # Invert x-axis to show decreasing samples (left to right)
    ax.invert_xaxis()
    
    # Style spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
    
    plt.title(title, fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    filename = title.replace(" ", "_").lower() + '.pdf'
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Optimal comparison plot saved to {filepath}")
    plt.close()


def plot_curvature_bounds(curvature_history: dict, title: str = "Curvature Bounds Evolution"):
    """
    Plot the evolution of the *restricted* curvature bounds during sample
    removal, matching Figure 25 / Section M ("Experimental Validation of
    Concavity") of the paper: at each removal step k, lambda_min/lambda_max
    are the spectral bounds of the Hessian submatrix restricted to R_k =
    T ∩ S^(k) (this step's removal *and* every later one still present), not
    just the diagonal entry of the single point removed at that step -- see
    calculate_restricted_curvature_bounds. These correspond directly to
    -M_R and -m_R in Lemma 5: for the RSC/RSM guarantee of Theorem 1 to
    hold, both curves should stay negative (concave) throughout.

    Args:
        curvature_history: Dictionary with keys 'lambda_min', 'lambda_max',
            'm' (single-point diagonal, kept for reference), and 'n_samples'.
        title: Plot title
    """
    if not curvature_history or not curvature_history.get('lambda_max'):
        print("No curvature history to plot")
        return

    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    # Disable LaTeX to avoid requiring dvipng/latex installation
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})

    # Create single plot
    fig, ax = plt.subplots(1, 1,
                          figsize=(figsizes.iclr2023(nrows=1, ncols=1, height_to_width_ratio=0.5)['figure.figsize']))

    color_min = "#1D6996"  # Blue
    color_max = "#CC503E"  # Red

    n_samples = np.array(curvature_history['n_samples'])
    lambda_min = np.array(curvature_history['lambda_min'])
    lambda_max = np.array(curvature_history['lambda_max'])

    # Debug output
    print(f"Plotting restricted curvature with {len(n_samples)} points")
    print(f"n_samples range: {min(n_samples)} to {max(n_samples)}")
    print(f"lambda_min range: {lambda_min.min()} to {lambda_min.max()}")
    print(f"lambda_max range: {lambda_max.min()} to {lambda_max.max()}")

    # Plot lambda_min and lambda_max of the R_k-restricted submatrix
    ax.plot(n_samples, lambda_min, marker='o', markersize=4,
            color=color_min, linewidth=1.5, alpha=0.9, label=r'$\lambda_{\min}(H_R)$')
    ax.plot(n_samples, lambda_max, marker='s', markersize=4,
            color=color_max, linewidth=1.5, alpha=0.9, label=r'$\lambda_{\max}(H_R)$')
    ax.set_xlabel(r'Number of Samples $|\mathcal{S}^{(k)}|$')
    ax.set_ylabel(r'Restricted Curvature')
    ax.yaxis.set_label_coords(-0.075, 0.5)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.legend(loc='best', fontsize=8)

    # Add horizontal line at y=0 to separate concave/convex
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)

    # Determine plot limits for shading, based on both curves
    all_vals = np.concatenate([lambda_min, lambda_max])
    v_min = all_vals.min()
    v_max = all_vals.max()

    # Ensure we have some margin even for small values
    if abs(v_max - v_min) < 1e-10:
        y_margin = 1.0  # Default margin if all values are the same
        y_bottom = v_min - y_margin
        y_top = v_max + y_margin
    else:
        y_margin = max(abs(v_max), abs(v_min)) * 0.2
        if y_margin < 0.1:  # Ensure minimum visible margin
            y_margin = 0.1
        y_bottom = v_min - y_margin
        y_top = v_max + y_margin

    # Fill regions: negative = concave (green), positive = convex (red).
    # Concavity of the *entire* submatrix (Lemma 5's RSC guarantee) requires
    # lambda_max <= 0, i.e. the worst-case (largest) eigenvalue, not just the
    # best case, so the shading is driven by lambda_max rather than lambda_min.
    if v_min < 0:
        ax.axhspan(y_bottom, 0, color='green', alpha=0.15, zorder=0)

    ax.axhspan(0, y_top, color='red', alpha=0.15, zorder=0)

    ax.set_ylim(y_bottom, y_top)
    ax.invert_xaxis()  # Invert x-axis to show decreasing samples

    # Add text labels at left boundary
    x_left = max(n_samples)  # Left boundary after inversion
    if v_min < 0:
        # Place "concave" text in the middle of the negative region
        y_concave = 0 + (0 - y_top) * 0.5
        ax.text(x_left, y_concave, 'concave',
                verticalalignment='center', horizontalalignment='left',
                fontsize=9, color='darkgreen', weight='bold')

    # Place "convex" text in the middle of the positive region
    y_convex = 0 + (y_top - 0) * 0.5
    ax.text(x_left, y_convex, 'convex',
            verticalalignment='center', horizontalalignment='left',
            fontsize=9, color='darkred', weight='bold')

    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    plt.title(title, fontsize=12)
    plt.tight_layout()

    # Save the plot
    filename = title.replace(" ", "_").lower() + '.pdf'
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)

    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Curvature bounds plot saved to {filepath}")
    plt.close()


def plot_curvature_condition_comparison(
    curvature_histories: dict,
    title: str = "Curvature Condition Comparison",
    y_limits: tuple = None,
):
    """Overlay lambda_min(H_R)/lambda_max(H_R) curves from multiple runs.

    Intended for a Gamma noise-prior concentration `a` ablation (Section M /
    Figure 25 of the paper): pass one curvature_history per `a` value (as
    returned by sequentially_optimize_samples / evaluate(..., return_plot_
    data=True)['curvature_history'], already computed via calculate_
    restricted_curvature_bounds's R_k = T ∩ S^(k) submatrix -- see plot_
    curvature_bounds), and this plots all of their lambda_min/lambda_max
    traces together so the effect of `a` on the RSC/RSM curvature condition
    (Assumption 1 / Lemma 5) is directly visible: larger `a` should make both
    curves settle further below zero (more concave / more stable).

    Args:
        curvature_histories: Dict mapping label (e.g. "a=2") -> curvature_history
            dict with keys 'n_samples', 'lambda_min', 'lambda_max'.
        title: Plot title / output filename stem.
        y_limits: Optional fixed y-axis limits as (ymin, ymax).
    """
    if not curvature_histories:
        print("No curvature histories provided")
        return

    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": False})
    plt.rcParams.update({"legend.frameon": False})

    # A full (not paper-column-fraction) base size, plus extra height for a
    # below-axes legend that can hold up to 2 entries (lambda_min/lambda_max)
    # per curve: rel_width=1/3 (this plot's original sizing) is only wide
    # enough for the title/tick labels at this theme's font sizes, and with
    # no layout engine reserving room for an external legend, that overlaps
    # the axes/title instead of being placed outside them.
    n_runs = max(len(curvature_histories), 1)
    legend_rows = -(-2 * n_runs // 4)  # ceil(2*n_runs / 4 columns)
    base_figsize = figsizes.iclr2023(nrows=1, ncols=1, height_to_width_ratio=0.62)['figure.figsize']
    fig, ax = plt.subplots(figsize=(base_figsize[0], base_figsize[1] + 0.22 * legend_rows))

    palette = [
        "#5F4690", "#1D6996", "#38A6A5", "#0F8554", "#73AF48",
        "#EDAD08", "#CC503E", "#94346E", "#6F4070", "#994E95", "#666666",
    ]

    global_y_min = None
    global_y_max = None

    for idx, (label, ch) in enumerate(curvature_histories.items()):
        if not ch or not ch.get('lambda_max'):
            continue
        n_samples = np.array(ch['n_samples'])
        lam_min = np.array(ch['lambda_min'])
        lam_max = np.array(ch['lambda_max'])
        if len(n_samples) == 0:
            continue

        color = palette[idx % len(palette)]
        # lambda_min: solid line; lambda_max: dashed line; shaded band between
        ax.plot(n_samples, lam_min, marker='o', markersize=3, linewidth=1.5,
                alpha=0.9, color=color, linestyle='-',
                label=rf'$\lambda_{{\min}}$ {label}')
        ax.plot(n_samples, lam_max, marker='s', markersize=3, linewidth=1.5,
                alpha=0.6, color=color, linestyle='--',
                label=rf'$\lambda_{{\max}}$ {label}')
        ax.fill_between(n_samples, lam_min, lam_max, color=color, alpha=0.07)

        with np.errstate(all='ignore'):
            loc_min = np.nanmin(lam_min)
            loc_max = np.nanmax(lam_max)
        if not np.isnan(loc_min):
            global_y_min = loc_min if global_y_min is None else min(global_y_min, loc_min)
        if not np.isnan(loc_max):
            global_y_max = loc_max if global_y_max is None else max(global_y_max, loc_max)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.set_xlabel(r'Number of Samples $|\mathcal{S}^{(k)}|$')
    ax.set_ylabel(r'Restricted Curvature')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.invert_xaxis()

    y_bottom = None
    y_top = None
    if y_limits is not None:
        y_bottom, y_top = y_limits
        ax.set_ylim(y_bottom, y_top)
    elif global_y_min is not None and global_y_max is not None:
        span = global_y_max - global_y_min
        margin = max(abs(global_y_min), abs(global_y_max)) * 0.15 if span > 1e-10 else 1.0
        margin = max(margin, 0.1)
        y_bottom = global_y_min - margin
        y_top = global_y_max + margin
        ax.set_ylim(y_bottom, y_top)

    if y_bottom is not None and y_top is not None:
        if y_bottom < 0:
            ax.axhspan(y_bottom, 0, color='green', alpha=0.15, zorder=0)
        ax.axhspan(0, y_top, color='red', alpha=0.15, zorder=0)

        # Axes-fraction placement (ax.transAxes), not data coordinates: a
        # data-coordinate anchor at the extreme edge of the plotted x-range
        # (e.g. global min n_samples) sits flush with -- and, with usetex
        # font metrics, can render past -- the axes' right border, clipping
        # the label in the saved figure. Anchoring in the axes' own [0, 1]
        # fraction space keeps 'convex'/'concave' just inside the panel
        # regardless of each run's actual x/y data range. Kept near the
        # right edge (top-/bottom-right corner) since lambda_min/lambda_max
        # converge toward zero there, leaving the corner uncluttered.
        ax.text(0.96, 0.94, 'convex', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='right',
                fontsize=7, color='darkred', weight='bold')
        if y_bottom < 0:
            ax.text(0.96, 0.06, 'concave', transform=ax.transAxes,
                    verticalalignment='bottom', horizontalalignment='right',
                    fontsize=7, color='darkgreen', weight='bold')

    # Below-axes legend (not to the right): with up to 2*n_runs entries this
    # avoids needing to widen the figure -- it just wraps into more rows,
    # which the figsize above already reserved height for.
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4,
                        fontsize=6, handlelength=1.2, handletextpad=0.4,
                        columnspacing=1.0, borderpad=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)

    ax.set_title(title, fontsize=10)

    filename = title.replace(" ", "_").lower() + '_condition.pdf'
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', bbox_extra_artists=(legend,), dpi=300)
    print(f"Curvature condition plot saved to {filepath}")
    plt.close(fig)


def plot_curvature_condition_grid(
    panels,
    filename_stem: str = "curvature_ablation_grid",
    y_limits: tuple = None,
):
    """Side-by-side grid version of plot_curvature_condition_comparison: one
    lambda_min/lambda_max-vs-|S^(k)| subplot per curvature ablation (e.g.
    symmetric / asymmetric / twitter), sharing a single legend below the
    figure instead of each panel repeating its own.

    Built to combine the three Gamma noise-prior concentration `a` sweeps
    (Section M / Figure 25 of the paper) run across the sin_symmetric,
    sin_asymmetric and twitter curvature-ablation configs into one figure --
    see run_curvature_ablations.py.

    Args:
        panels: List of (short_title, curvature_histories) pairs, left to
            right. curvature_histories maps label (e.g. "a=2") -> dict with
            'n_samples', 'lambda_min', 'lambda_max' (as returned by
            evaluate(..., return_plot_data=True)['curvature_history']).
        filename_stem: Output filename stem (saved as <stem>.pdf under the
            figures dir).
        y_limits: Optional (ymin, ymax) applied to every subplot. None (the
            default) auto-scales each subplot to its own data -- the three
            ablations differ by roughly an order of magnitude in curvature
            and sample count, so a shared y-axis would flatten the smaller
            two into unreadable slivers.
    """
    panels = [(panel_title, ch) for panel_title, ch in panels if ch]
    if not panels:
        print("No curvature histories provided")
        return

    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"legend.frameon": False})

    n_panels = len(panels)

    # Stable color per gamma_prior_a label, shared across panels so e.g.
    # "a=5" is the same color in every subplot -- required for the panels'
    # legend entries to collapse into one shared legend.
    all_labels = []
    for _, curvature_histories in panels:
        for label in curvature_histories:
            if label not in all_labels:
                all_labels.append(label)

    palette = [
        "#5F4690", "#1D6996", "#38A6A5", "#0F8554", "#73AF48",
        "#EDAD08", "#CC503E", "#94346E", "#6F4070", "#994E95", "#666666",
    ]
    label_color = {label: palette[i % len(palette)] for i, label in enumerate(all_labels)}

    # Reserve room below the axes for the shared legend, same sizing logic
    # as the single-panel version (2 entries -- lambda_min/lambda_max --
    # per gamma_prior_a value, wrapped at LEGEND_NCOL columns).
    LEGEND_NCOL = 6
    n_runs = max(len(all_labels), 1)
    legend_rows = -(-2 * n_runs // LEGEND_NCOL)  # ceil(2*n_runs / LEGEND_NCOL columns)
    base_figsize = figsizes.iclr2023(nrows=1, ncols=n_panels, height_to_width_ratio=0.62)['figure.figsize']
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(base_figsize[0], base_figsize[1] + 0.22 * legend_rows),
        sharey=False,
    )
    axes = np.atleast_1d(axes)

    legend_handles = {}

    for ax, (panel_title, curvature_histories) in zip(axes, panels):
        global_y_min = None
        global_y_max = None

        for label, ch in curvature_histories.items():
            if not ch or not ch.get('lambda_max'):
                continue
            n_samples = np.array(ch['n_samples'])
            lam_min = np.array(ch['lambda_min'])
            lam_max = np.array(ch['lambda_max'])
            if len(n_samples) == 0:
                continue

            color = label_color[label]
            label_min = rf'$\lambda_{{\min}}$ {label}'
            label_max = rf'$\lambda_{{\max}}$ {label}'
            line_min, = ax.plot(n_samples, lam_min, marker='o', markersize=3, linewidth=1.5,
                                 alpha=0.9, color=color, linestyle='-', label=label_min)
            line_max, = ax.plot(n_samples, lam_max, marker='s', markersize=3, linewidth=1.5,
                                 alpha=0.6, color=color, linestyle='--', label=label_max)
            ax.fill_between(n_samples, lam_min, lam_max, color=color, alpha=0.07)

            legend_handles.setdefault(label_min, line_min)
            legend_handles.setdefault(label_max, line_max)

            with np.errstate(all='ignore'):
                loc_min = np.nanmin(lam_min)
                loc_max = np.nanmax(lam_max)
            if not np.isnan(loc_min):
                global_y_min = loc_min if global_y_min is None else min(global_y_min, loc_min)
            if not np.isnan(loc_max):
                global_y_max = loc_max if global_y_max is None else max(global_y_max, loc_max)

        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.0, alpha=0.7)
        ax.set_xlabel(r'Number of Samples $|\mathcal{S}^{(k)}|$')
        ax.grid(True, alpha=0.3, linewidth=0.5)
        ax.invert_xaxis()

        y_bottom = None
        y_top = None
        if y_limits is not None:
            y_bottom, y_top = y_limits
            ax.set_ylim(y_bottom, y_top)
        elif global_y_min is not None and global_y_max is not None:
            span = global_y_max - global_y_min
            margin = max(abs(global_y_min), abs(global_y_max)) * 0.15 if span > 1e-10 else 1.0
            margin = max(margin, 0.1)
            y_bottom = global_y_min - margin
            y_top = global_y_max + margin

            # Guarantee the 'convex' label a clear top strip regardless of
            # how the data happens to be shaped: when the positive part of
            # the range is a tiny sliver hugging zero (e.g. twitter,
            # asymmetric -- curves converge to near-zero at the right edge,
            # where the label also sits), the margin above only pads
            # global_y_max a little, so the zero-line ends up high in the
            # axes and the label collides with the converging curves. Widen
            # y_top (never shrink it) so the zero-line sits at or below
            # TOP_LABEL_CLEARANCE of the axes height, reserving real
            # headroom above it for the label. A no-op when the margin
            # above already clears that bar (e.g. symmetric).
            TOP_LABEL_CLEARANCE = 0.75
            below_zero = -y_bottom
            min_y_top_for_label = below_zero * (1 - TOP_LABEL_CLEARANCE) / TOP_LABEL_CLEARANCE
            y_top = max(y_top, min_y_top_for_label)
            ax.set_ylim(y_bottom, y_top)

        if y_bottom is not None and y_top is not None:
            if y_bottom < 0:
                ax.axhspan(y_bottom, 0, color='green', alpha=0.15, zorder=0)
            ax.axhspan(0, y_top, color='red', alpha=0.15, zorder=0)

            # Axes-fraction placement (ax.transAxes): each panel has its own
            # x/y data range (the three ablations differ by roughly an order
            # of magnitude), so anchoring in data coordinates at the extreme
            # edge of that range -- as the single-panel version originally
            # did -- sits flush with the axes border and can render past it.
            # Fraction coordinates keep 'convex'/'concave' just inside every
            # panel regardless of its own range. Kept near the right edge
            # (top-/bottom-right corner), where lambda_min/lambda_max
            # converge toward zero and the corner stays uncluttered. Shifted
            # up (0.97, was 0.94) now that the TOP_LABEL_CLEARANCE headroom
            # above is guaranteed, so it can't run back into the curves.
            ax.text(0.96, 0.97, 'convex', transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    fontsize=7, color='darkred', weight='bold')
            if y_bottom < 0:
                ax.text(0.96, 0.06, 'concave', transform=ax.transAxes,
                        verticalalignment='bottom', horizontalalignment='right',
                        fontsize=7, color='darkgreen', weight='bold')

        for spine in ax.spines.values():
            spine.set_linewidth(1.25)

        ax.set_title(panel_title, fontsize=10)

    axes[0].set_ylabel(r'Restricted Curvature')

    # One shared legend below all subplots instead of each panel repeating
    # its own -- entries are deduplicated by label, so identical
    # gamma_prior_a values across panels collapse to a single entry.
    legend_labels = list(legend_handles.keys())
    ncol = min(LEGEND_NCOL, len(legend_labels)) if legend_labels else 1
    legend = fig.legend([legend_handles[l] for l in legend_labels], legend_labels,
                         loc='upper center', bbox_to_anchor=(0.5, 0.02), ncol=ncol,
                         fontsize=6, handlelength=1.2, handletextpad=0.4,
                         columnspacing=1.0, borderpad=0.4)

    filename = filename_stem.replace(" ", "_").lower() + '.pdf'
    figures_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    filepath = os.path.join(figures_dir, filename)
    fig.savefig(filepath, bbox_inches='tight', bbox_extra_artists=(legend,), dpi=300)
    print(f"Curvature condition grid plot saved to {filepath}")
    plt.close(fig)


