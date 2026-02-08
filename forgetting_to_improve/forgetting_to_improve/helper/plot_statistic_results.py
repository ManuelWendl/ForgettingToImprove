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
        'none_joint': '#666666',
        'targetSampling_joint': '#73AF48',
        'relevance_pursuit_joint': '#CC503E',
        'warped_gp_joint': '#38A6A5',
        'heteroscedastic_gp_joint': '#1D6996'
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
        basename = os.path.basename(filepath)
        # e.g., "twitter_fitting_comparison_method_noise_20260128_161421.txt"
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
        'yacht': 'Yacht Hydrodynamics'
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
