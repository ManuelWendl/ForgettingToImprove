import os
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from tueplots import bundles, figsizes
from seaborn import axes_style


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    
    This function creates a RadarAxes projection and registers it.
    
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


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


def plot_radar_comparison(results_folder, output_folder=None):
    """
    Create radar plots comparing different methods across benchmarks.
    
    Parameters
    ----------
    results_folder : str
        Path to folder containing results .txt files
    output_folder : str, optional
        Path to save output figures. If None, saves to figures/radar_plots/
    """
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
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
    
    # Find all .txt files in the results folder
    txt_files = []
    for fname in os.listdir(results_folder):
        if fname.endswith('.txt'):
            txt_files.append(os.path.join(results_folder, fname))
    
    if not txt_files:
        print(f"No .txt files found in {results_folder}")
        return
    
    # Parse all files
    benchmarks = {}
    for filepath in txt_files:
        # Extract benchmark name from filename
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # e.g., "twitter_fitting_comparison_method_noise_20260128_161421"
        benchmark_name = basename.split('_fitting_comparison')[0]
        
        results = parse_results_file(filepath)
        if results:
            benchmarks[benchmark_name] = results
    
    if not benchmarks:
        print("No valid results found in files")
        return
    
    # Benchmark display name mapping
    benchmark_display_names = {
        'twitter': 'Dow Jones',
        'boston': 'Boston Housing',
        'casp': 'CASP',
        'yacht': 'Yacht'
    }
    
    # Create radar factory
    num_metrics = len(metrics)
    theta = radar_factory(num_metrics, frame='polygon')
    
    # Determine grid size
    n_benchmarks = len(benchmarks)
    n_cols = min(3, n_benchmarks)
    n_rows = (n_benchmarks + n_cols - 1) // n_cols
    
    # Create figure
    fig = plt.figure(figsize=(figsizes.neurips2024(nrows=n_rows, ncols=n_cols, 
                                                     height_to_width_ratio=1.0)['figure.figsize']))
    
    # Plot each benchmark
    for idx, (benchmark_name, data) in enumerate(sorted(benchmarks.items())):
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='radar')
        
        # Normalize metrics (only considering methods we're plotting)
        normalized_data = normalize_metrics(data, metrics, methods_to_include=list(method_mapping.keys()))
        
        # Plot all methods except sequential_joint first (so it appears on top)
        for method_key in method_mapping.keys():
            if method_key == 'sequential_joint':
                continue  # Plot this last
            if method_key in normalized_data:
                values = [normalized_data[method_key].get(m, 0.0) for m in metrics]
                # Close the radar line by appending the first value
                values_closed = values + [values[0]]
                theta_closed = np.append(theta, theta[0])
                ax.plot(theta_closed, values_closed, 
                       color=method_colors[method_key],
                       linewidth=1.0,
                       label=method_mapping[method_key],
                       alpha=1.0)
        
        # Plot sequential_joint last so it appears on top
        if 'sequential_joint' in normalized_data:
            values = [normalized_data['sequential_joint'].get(m, 0.0) for m in metrics]
            # Close the radar line by appending the first value
            values_closed = values + [values[0]]
            theta_closed = np.append(theta, theta[0])
            ax.plot(theta_closed, values_closed, 
                   color=method_colors['sequential_joint'],
                   linewidth=2,
                   label=method_mapping['sequential_joint'],
                   alpha=1.0,
                   zorder=10)  # Higher zorder to ensure it's on top
        
        # Find best method for each metric and color labels accordingly
        # Only consider methods that are being plotted
        metric_colors = []
        for metric in metrics:
            best_method = None
            best_value = -1
            for method_key in method_mapping.keys():  # Only check plotted methods
                if method_key in normalized_data and metric in normalized_data[method_key]:
                    val = normalized_data[method_key][metric]
                    if val > best_value:
                        best_value = val
                        best_method = method_key
            
            # Use the color of the best method for this metric
            if best_method and best_method in method_colors:
                metric_colors.append(method_colors[best_method])
            else:
                metric_colors.append('black')  # Fallback color
        
        # Set variable labels with colors
        ax.set_varlabels(metrics)
        # Color each label
        for label, color in zip(ax.get_xticklabels(), metric_colors):
            label.set_color(color)
        
        # Set title with mapped display name
        display_name = benchmark_display_names.get(benchmark_name, benchmark_name.replace('_', ' ').title())
        ax.set_title(display_name,
                    weight='bold', size=10, position=(0.5, 1.1),
                    horizontalalignment='center', verticalalignment='center')
        
        # Configure radial limits - extend slightly beyond 1 to prevent clipping at edges
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 1.0])
        ax.set_yticklabels(['0', '1'], fontsize=7)
        
        # Grid styling
        ax.grid(True, linewidth=0.5, color='gainsboro', alpha=0.7)
    
    # Create shared legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        # Reorder so Forgetting to Improve appears first in legend
        # Find the index of 'Forgetting to Improve' in labels
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
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    # Save figure
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'radar_plots')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, 'benchmark_radar_comparison.pdf')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Radar plot saved to {output_path}")
    plt.close()


def plot_line_profile_comparison(results_folder, output_folder=None):
    """
    Create line profile plots comparing different methods across benchmarks.
    
    Parameters
    ----------
    results_folder : str
        Path to folder containing results .txt files
    output_folder : str, optional
        Path to save output figures. If None, saves to figures/radar_plots/
    """
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
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
    
    # Line styles for better distinction
    method_linestyles = {
        'sequential_joint': '-',
        'none_joint': '-',
        'targetSampling_joint': '-',
        'relevance_pursuit_joint': '-',
        'warped_gp_joint': '-',
        'heteroscedastic_gp_joint': '-'
    }
    
    # Markers for each method
    method_markers = {
        'sequential_joint': 'o',
        'none_joint': 's',
        'targetSampling_joint': '^',
        'relevance_pursuit_joint': 'D',
        'warped_gp_joint': 'v',
        'heteroscedastic_gp_joint': 'p'
    }
    
    # Metrics to plot
    metrics = ['MSE', 'MAE', 'NLL', 'CE', 'V-ECE', 'Sharpness']
    
    # Find all .txt files in the results folder
    txt_files = []
    for fname in os.listdir(results_folder):
        if fname.endswith('.txt'):
            txt_files.append(os.path.join(results_folder, fname))
    
    if not txt_files:
        print(f"No .txt files found in {results_folder}")
        return
    
    # Parse all files
    benchmarks = {}
    for filepath in txt_files:
        # Extract benchmark name from filename
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # e.g., "twitter_fitting_comparison_method_noise_20260128_161421"
        benchmark_name = basename.split('_fitting_comparison')[0]
        
        results = parse_results_file(filepath)
        if results:
            benchmarks[benchmark_name] = results
    
    if not benchmarks:
        print("No valid results found in files")
        return
    
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
                                                          height_to_width_ratio=0.8)['figure.figsize'])
    
    # Ensure axes is always a list
    if n_benchmarks == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_benchmarks > 1 else [axes]
    
    # Plot each benchmark
    for idx, (benchmark_name, data) in enumerate(sorted(benchmarks.items())):
        ax = axes[idx]
        
        # Normalize metrics (only considering methods we're plotting)
        normalized_data = normalize_metrics(data, metrics, methods_to_include=list(method_mapping.keys()))
        
        # X-axis positions for metrics
        x_positions = np.arange(len(metrics))
        
        # Find best method for each metric
        best_methods = []
        for metric in metrics:
            best_method = None
            best_value = -1
            for method_key in method_mapping.keys():
                if method_key in normalized_data and metric in normalized_data[method_key]:
                    val = normalized_data[method_key][metric]
                    if val > best_value:
                        best_value = val
                        best_method = method_key
            best_methods.append(best_method)
        
        # Plot all methods except sequential_joint first (so it appears on top)
        for method_key in method_mapping.keys():
            if method_key == 'sequential_joint':
                continue  # Plot this last
            if method_key in normalized_data:
                values = [normalized_data[method_key].get(m, 0.0) for m in metrics]
                ax.plot(x_positions, values, 
                       color=method_colors[method_key],
                       linewidth=1.5,
                       linestyle=method_linestyles[method_key],
                       marker='none',
                       label=method_mapping[method_key],
                       alpha=0.8)
                
                # Add markers only for best values
                for i, (x, val, metric) in enumerate(zip(x_positions, values, metrics)):
                    if best_methods[i] == method_key:
                        ax.plot(x, val, 
                               marker=method_markers[method_key],
                               markersize=5,
                               color=method_colors[method_key],
                               zorder=5)
        
        # Plot sequential_joint last so it appears on top
        if 'sequential_joint' in normalized_data:
            values = [normalized_data['sequential_joint'].get(m, 0.0) for m in metrics]
            ax.plot(x_positions, values, 
                   color=method_colors['sequential_joint'],
                   linewidth=2.5,
                   linestyle=method_linestyles['sequential_joint'],
                   marker='none',
                   label=method_mapping['sequential_joint'],
                   alpha=1.0,
                   zorder=10)  # Higher zorder to ensure it's on top
            
            # Add markers only for best values
            for i, (x, val, metric) in enumerate(zip(x_positions, values, metrics)):
                if best_methods[i] == 'sequential_joint':
                    ax.plot(x, val, 
                           marker=method_markers['sequential_joint'],
                           markersize=6,
                           color=method_colors['sequential_joint'],
                           zorder=15)
        
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
        
        ax.set_ylim(-0.05, 1.1)
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
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'radar_plots')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, 'benchmark_line_profile_comparison.pdf')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Line profile plot saved to {output_path}")
    plt.close()


def plot_bar_comparison(results_folder, output_folder=None):
    """
    Create bar plots with dodged/grouped bars comparing different methods across benchmarks.
    
    Parameters
    ----------
    results_folder : str
        Path to folder containing results .txt files
    output_folder : str, optional
        Path to save output figures. If None, saves to figures/radar_plots/
    """
    # Apply ICLR 2023 styling
    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
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
    
    # Find all .txt files in the results folder
    txt_files = []
    for fname in os.listdir(results_folder):
        if fname.endswith('.txt'):
            txt_files.append(os.path.join(results_folder, fname))
    
    if not txt_files:
        print(f"No .txt files found in {results_folder}")
        return
    
    # Parse all files
    benchmarks = {}
    for filepath in txt_files:
        # Extract benchmark name from filename
        basename = os.path.splitext(os.path.basename(filepath))[0]
        # e.g., "twitter_fitting_comparison_method_noise_20260128_161421"
        benchmark_name = basename.split('_fitting_comparison')[0]
        
        results = parse_results_file(filepath)
        if results:
            benchmarks[benchmark_name] = results
    
    if not benchmarks:
        print("No valid results found in files")
        return
    
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
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures', 'radar_plots')
    os.makedirs(output_folder, exist_ok=True)
    
    output_path = os.path.join(output_folder, 'benchmark_bar_comparison.pdf')
    fig.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Bar plot saved to {output_path}")
    plt.close()


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        results_folder = sys.argv[1]
    else:
        # Default to stable_baselines folder
        script_dir = os.path.dirname(os.path.dirname(__file__))
        results_folder = os.path.join(script_dir, 'results', 'stable_baselines')
    
    print(f"Plotting results from: {results_folder}")
    plot_radar_comparison(results_folder)
    plot_line_profile_comparison(results_folder)
    plot_bar_comparison(results_folder)


def _latex_p(p):
    """p-value as LaTeX math, e.g. 'p = 1.1 \\cdot 10^{-5}'."""
    if not np.isfinite(p):
        return 'p = \\mathrm{n/a}'
    if p >= 1e-3:
        return f'p = {p:.2g}'
    mant, exp = f'{p:.1e}'.split('e')
    return f'p = {mant} \\cdot 10^{{{int(exp)}}}'


def plot_noise_ablation_paired(results_folder, benchmarks, metrics=('RMSE', 'NLL'),
                               noise_a='joint', noise_b='aleatoric', output_folder=None,
                               filename='noise_ablation_paired.pdf'):
    """Per-seed paired scatter of two removal criteria (x: noise_a, y: noise_b).

    One panel per (metric, benchmark); one dot per seed. Seeds where both
    criteria give the identical model sit on the diagonal in gray; seeds that
    differ are colored. Points above y = x are seeds where noise_a is better
    (all metrics here are lower-is-better). Each panel reports the win counts
    and the paired Wilcoxon p-value over the differing seeds.

    Args:
        results_folder: Folder with per-seed '<benchmark>_<noise>.json' files
            ({seed: {metric: value}}).
        benchmarks: List of (panel title, file stem) pairs.
        metrics: Metrics to plot, one row each.
        noise_a, noise_b: The two criteria to compare (x and y axis).
        output_folder: Where to save (defaults to the package figures dir).
        filename: Output file name.
    """
    import json
    from scipy.stats import wilcoxon

    theme = bundles.neurips2024()
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(theme)
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": True})
    plt.rcParams.update({"legend.frameon": False})

    color_diff = "#1D6996"
    color_same = "#B0B0B0"
    labels = {'joint': 'joint', 'aleatoric': 'aleatoric only', 'aleatoric_scaled': 'aleatoric only'}

    n_rows, n_cols = len(metrics), len(benchmarks)
    figsize = figsizes.neurips2024(nrows=n_rows, ncols=n_cols, height_to_width_ratio=1.0)['figure.figsize']
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for col, (title, stem) in enumerate(benchmarks):
        ra = json.load(open(os.path.join(results_folder, f"{stem}_{noise_a}.json")))
        rb = json.load(open(os.path.join(results_folder, f"{stem}_{noise_b}.json")))
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


def _latex_p_short(p):
    """Compact p-value for narrow panels: 'p = 0.014' or 'p < 10^{-k}'."""
    if not np.isfinite(p):
        return 'p = \\mathrm{n/a}'
    if p >= 1e-3:
        return f'p = {p:.2g}'
    return f'p < 10^{{{int(np.ceil(np.log10(p)))}}}'


def _noise_ablation_panels(results_folder, stem, num_train_samples, metrics, noise_a, noise_b):
    """Per-seed values for each panel: [(key, label, values_a, values_b)]."""
    import json
    ra = json.load(open(os.path.join(results_folder, f"{stem}_{noise_a}.json")))
    rb = json.load(open(os.path.join(results_folder, f"{stem}_{noise_b}.json")))
    seeds = sorted(set(ra) & set(rb), key=int)
    panels = []
    for m in metrics:
        panels.append((m, m, np.array([ra[s][m] for s in seeds]), np.array([rb[s][m] for s in seeds])))
    panels.append(('removed', r'\# removed',
                   np.array([num_train_samples - ra[s]['n_opt'] for s in seeds], dtype=float),
                   np.array([num_train_samples - rb[s]['n_opt'] for s in seeds], dtype=float)))
    return panels


def _draw_noise_bar_panel(ax, label, va, vb, colors, tick_labels):
    """Two bars: the mean over seeds of each criterion."""
    means = [va.mean(), vb.mean()]
    ax.bar([0, 1], means, width=0.7, color=colors, zorder=2)
    ax.axhline(0, color='black', linewidth=0.6)
    lo, hi = min(0.0, *means), max(0.0, *means)
    span = hi - lo
    ax.set_ylim(lo - (0.1 * span if lo < 0 else 0.0), hi + 0.1 * span)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_title(label)
    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def _apply_noise_ablation_style(rel_width):
    plt.rcParams.update(axes_style("white"))
    plt.rcParams.update(bundles.neurips2024(rel_width=rel_width))
    plt.rcParams.update({"text.latex.preamble": r"\usepackage{amsmath}\usepackage{times}"})
    plt.rcParams.update({"text.usetex": True})


_NOISE_COLORS = ("#1D6996", "#94346E")
_NOISE_NAMES = {'joint': 'joint', 'aleatoric': 'aleatoric', 'aleatoric_scaled': 'aleatoric'}


def plot_noise_ablation_bars(results_folder, stem, num_train_samples, title=None,
                             metrics=('RMSE', 'NLL'), noise_a='joint', noise_b='aleatoric',
                             output_folder=None, filename=None):
    """Bar plot (mean over seeds) comparing two removal criteria on one
    benchmark: one panel per metric plus one for the number of removed
    samples, two bars each.

    Args:
        results_folder: Folder with per-seed '<stem>_<noise>.json' files
            ({seed: {metric: value, 'n_opt': retained samples}}).
        stem: Benchmark file stem.
        num_train_samples: Initial training-set size (removed = this - n_opt).
        title: Figure title.
        metrics: Metrics to plot (lower is better).
        noise_a, noise_b: The two criteria to compare.
        output_folder: Where to save (defaults to the package figures dir).
        filename: Output file name (default 'noise_ablation_bars_<stem>.pdf').
    """
    _apply_noise_ablation_style(0.5)
    panels = _noise_ablation_panels(results_folder, stem, num_train_samples, metrics, noise_a, noise_b)
    figsize = figsizes.neurips2024(rel_width=0.5, nrows=1, ncols=len(panels),
                                   height_to_width_ratio=1.6)['figure.figsize']
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    for ax, (_, label, va, vb) in zip(np.atleast_1d(axes), panels):
        _draw_noise_bar_panel(ax, label, va, vb, _NOISE_COLORS,
                              [_NOISE_NAMES[noise_a], _NOISE_NAMES[noise_b]])
    if title:
        fig.suptitle(title)

    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, filename or f"noise_ablation_bars_{stem}.pdf")
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    print(f"Noise ablation bar plot saved to {filepath}")
    plt.close(fig)


def plot_noise_ablation_bars_grid(results_folder, benchmarks, metrics=('RMSE', 'NLL'),
                                  noise_a='joint', noise_b='aleatoric', output_folder=None,
                                  filename='noise_ablation_bars.pdf'):
    """Joint vs aleatoric-only removal criterion, one titled subfigure per
    benchmark side by side in one full-text-width figure. Each subfigure has
    one small panel per quantity (metrics and number of removed samples, each
    on its own y-scale, named by its x-label) with one bar per criterion
    (mean over seeds), styled like plot_bar_comparison
    (Forgetting-to-Improve purple for joint) with one shared legend.

    Args:
        results_folder: Folder with per-seed '<stem>_<noise>.json' files.
        benchmarks: List of (title, stem, num_train_samples), left to right.
        metrics, noise_a, noise_b: As in plot_noise_ablation_bars.
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
    for subfig, (title, stem, num_train_samples) in zip(subfigs, benchmarks):
        panels = _noise_ablation_panels(results_folder, stem, num_train_samples, metrics, noise_a, noise_b)
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
