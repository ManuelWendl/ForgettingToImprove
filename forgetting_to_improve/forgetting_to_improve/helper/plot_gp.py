import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker
from tueplots import bundles, figsizes
from seaborn import axes_style
import os

def plot_predictions(objective_function, x, x_samples, y_samples, y_pred, y_std, x_limits, A_limits, title=None, annotation=False):
    """
    Plot the GP predictions along with the true function using ICLR 2023 style.
    """
    # Apply ICLR 2023 styling
    theme = bundles.iclr2023()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update(figsizes.iclr2023(nrows=1, ncols=2))  # Changed to ncols=2 for side-by-side
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
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
    box = Rectangle((A_limits[0], min(objective_function(x))-1), A_limits[1] - A_limits[0], max(objective_function(x))+1 - (min(objective_function(x))-1), linewidth=1.25, edgecolor='gray', 
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
    
    # Set y-limits
    ax.set_ylim(min(objective_function(x)), max(objective_function(x)))
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