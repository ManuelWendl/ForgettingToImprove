"""
Analyze Boston Housing dataset to identify good target regions.

This script loads the Boston dataset and analyzes it to find regions
with relatively smooth behavior and low noise for defining target regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('forgetting_to_improve/forgetting_to_improve')

from helper.data_loader import load_boston

def analyze_local_smoothness(x, y, n_neighbors=10):
    """
    Analyze local smoothness by looking at variance of neighbors.
    
    Parameters
    ----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
    n_neighbors : int
        Number of neighbors to consider
        
    Returns
    -------
    np.ndarray
        Local variance for each point
    """
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(x)
    
    # Calculate local variance for each point
    local_variance = np.zeros(len(y))
    for i in range(len(y)):
        neighbor_y = y[indices[i][1:]]  # Exclude the point itself
        local_variance[i] = np.var(neighbor_y)
    
    return local_variance

def find_smooth_regions(x, y, local_variance, percentile=25):
    """
    Find regions with low local variance (smooth regions).
    
    Parameters
    ----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
    local_variance : np.ndarray
        Local variance at each point
    percentile : float
        Percentile threshold for defining smooth regions
        
    Returns
    -------
    np.ndarray
        Boolean mask for smooth regions
    """
    threshold = np.percentile(local_variance, percentile)
    smooth_mask = local_variance < threshold
    return smooth_mask

def analyze_target_value_regions(y, n_bins=10):
    """
    Analyze distribution across target value ranges.
    
    Parameters
    ----------
    y : np.ndarray
        Target values
    n_bins : int
        Number of bins to divide the target range
        
    Returns
    -------
    tuple
        (bin_edges, counts, bin_centers)
    """
    counts, bin_edges = np.histogram(y, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, counts, bin_centers

def find_dimension_wise_bounds(x, smooth_mask, margin=0.05):
    """
    Find dimension-wise bounds for the smooth region.
    
    Parameters
    ----------
    x : np.ndarray
        Input features
    smooth_mask : np.ndarray
        Boolean mask for smooth points
    margin : float
        Margin to add to bounds (as fraction of range)
        
    Returns
    -------
    list
        List of [min, max] bounds for each dimension
    """
    x_smooth = x[smooth_mask]
    n_dims = x.shape[1]
    
    bounds = []
    for i in range(n_dims):
        dim_min = x_smooth[:, i].min()
        dim_max = x_smooth[:, i].max()
        dim_range = dim_max - dim_min
        
        # Add small margin
        dim_min -= margin * dim_range
        dim_max += margin * dim_range
        
        bounds.append([dim_min, dim_max])
    
    return bounds

def count_points_in_bounds(x, bounds):
    """Count how many points fall within the given bounds."""
    mask = np.ones(len(x), dtype=bool)
    for i, (lower, upper) in enumerate(bounds):
        mask &= (x[:, i] >= lower) & (x[:, i] <= upper)
    return mask.sum()

def main():
    print("Loading Boston Housing dataset...")
    x, y = load_boston()
    
    print(f"\nDataset statistics:")
    print(f"  Number of samples: {len(y)}")
    print(f"  Number of features: {x.shape[1]}")
    print(f"  Target (y) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Target (y) mean: {y.mean():.3f}, std: {y.std():.3f}")
    
    # Analyze local smoothness
    print("\nAnalyzing local smoothness...")
    local_variance = analyze_local_smoothness(x, y, n_neighbors=10)
    
    print(f"  Local variance range: [{local_variance.min():.4f}, {local_variance.max():.4f}]")
    print(f"  Local variance mean: {local_variance.mean():.4f}, std: {local_variance.std():.4f}")
    
    # Target: Find region with approximately 20 smooth points
    target_n_points = 20
    print(f"\n\nFinding dimension-wise bounds for ~{target_n_points} smoothest points...")
    
    # Sort by local variance and take the smoothest points
    sorted_indices = np.argsort(local_variance)
    
    # Try different numbers of points and different margin values to find best fit
    best_n_smooth = target_n_points
    best_bounds = None
    best_actual_count = 0
    best_margin = 0.05
    all_results = []
    
    for n_smooth in [8, 10, 12, 15, 18, 20]:
        for margin in [0.0, 0.02, 0.05, 0.08]:
            smooth_mask = np.zeros(len(x), dtype=bool)
            smooth_mask[sorted_indices[:n_smooth]] = True
            
            # Calculate dimension-wise bounds
            bounds = find_dimension_wise_bounds(x, smooth_mask, margin=margin)
            
            # Count how many points fall in these bounds
            actual_count = count_points_in_bounds(x, bounds)
            
            all_results.append((n_smooth, margin, bounds, actual_count))
            
            # Pick the one closest to target (initialize if first or better match)
            if best_bounds is None or abs(actual_count - target_n_points) < abs(best_actual_count - target_n_points):
                best_n_smooth = n_smooth
                best_bounds = bounds
                best_actual_count = actual_count
                best_margin = margin
    
    # Show top candidates
    print("\n  Top candidates for ~20 points:")
    all_results.sort(key=lambda x: abs(x[3] - target_n_points))
    for i, (n, m, b, count) in enumerate(all_results[:5]):
        print(f"    {i+1}. n={n:2d}, margin={m:.2f} -> {count:3d} points (var: {local_variance[sorted_indices[:n]].mean():.4f})")
    
    print(f"\n  BEST CHOICE: {best_n_smooth} smoothest points (margin={best_margin:.2f}) -> {best_actual_count} points in region")
    
    # Use the best bounds
    smooth_mask = np.zeros(len(x), dtype=bool)
    smooth_mask[sorted_indices[:best_n_smooth]] = True
    bounds = best_bounds
    
    # Get statistics for smooth region
    y_smooth = y[smooth_mask]
    print(f"\n  Smoothest {best_n_smooth} points statistics:")
    print(f"    Target range: [{y_smooth.min():.3f}, {y_smooth.max():.3f}]")
    print(f"    Target mean: {y_smooth.mean():.3f}, std: {y_smooth.std():.3f}")
    print(f"    Mean local variance: {local_variance[smooth_mask].mean():.4f}")
    
    # Print dimension-wise bounds
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    
    print(f"\n  Dimension-wise bounds (A_limits):")
    for i, (feature, (lower, upper)) in enumerate(zip(feature_names, bounds)):
        print(f"    {i+1:2d}. {feature:10s}: [{lower:7.3f}, {upper:7.3f}]  "
              f"(range: {upper-lower:.3f})")
    
    # Find smooth regions at different thresholds for comparison
    print("\n\nFor comparison - percentile-based regions:")
    for percentile in [10, 25, 50]:
        smooth_mask_pct = find_smooth_regions(x, y, local_variance, percentile=percentile)
        n_smooth = smooth_mask_pct.sum()
        y_smooth = y[smooth_mask_pct]
        print(f"\n  Smoothest {percentile}% of regions ({n_smooth} points):")
        print(f"    Target range: [{y_smooth.min():.3f}, {y_smooth.max():.3f}]")
        print(f"    Target mean: {y_smooth.mean():.3f}, std: {y_smooth.std():.3f}")
    
    # Analyze by target value bins
    print("\n\nAnalyzing by target value ranges...")
    bin_edges, counts, bin_centers = analyze_target_value_regions(y, n_bins=10)
    
    for i in range(len(bin_centers)):
        bin_mask = (y >= bin_edges[i]) & (y < bin_edges[i+1])
        bin_variance = local_variance[bin_mask]
        print(f"  Bin [{bin_edges[i]:.2f}, {bin_edges[i+1]:.2f}]: "
              f"{counts[i]:3d} points, avg local var: {bin_variance.mean():.4f}")
    
    # Print formatted YAML output for easy copy-paste
    print("\n\n" + "="*70)
    print("YAML CONFIG OUTPUT FOR boston.yaml")
    print("="*70)
    print(f"\n  # Target region with ~{best_actual_count} smooth points (selected from {best_n_smooth} smoothest)")
    print(f"  # Mean local variance: {local_variance[smooth_mask].mean():.4f}")
    print(f"  # Target value range: [{y_smooth.min():.3f}, {y_smooth.max():.3f}]")
    print("  A_limits: [")
    for i, (feature, (lower, upper)) in enumerate(zip(feature_names, bounds)):
        print(f"    [{lower:7.3f}, {upper:7.3f}],  # {feature}")
    print("  ]")
    print("="*70)
    
    # Create visualizations
    print("\n\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Use the smooth mask for the best selection
    smooth_mask_viz = np.zeros(len(x), dtype=bool)
    smooth_mask_viz[sorted_indices[:best_n_smooth]] = True
    
    # Also create a mask for points within the bounding box
    bbox_mask = np.ones(len(x), dtype=bool)
    for i, (lower, upper) in enumerate(bounds):
        bbox_mask &= (x[:, i] >= lower) & (x[:, i] <= upper)
    
    # 1. Target value distribution
    axes[0, 0].hist(y, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Target Value (normalized)', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].set_title('Target Value Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Local variance distribution
    axes[0, 1].hist(local_variance, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Local Variance', fontsize=12)
    axes[0, 1].set_ylabel('Count', fontsize=12)
    axes[0, 1].set_title('Local Variance Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Target vs Local Variance scatter
    axes[0, 2].scatter(y, local_variance, alpha=0.5, s=20)
    axes[0, 2].set_xlabel('Target Value (normalized)', fontsize=12)
    axes[0, 2].set_ylabel('Local Variance', fontsize=12)
    axes[0, 2].set_title('Target Value vs Local Variance', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Add horizontal line for 25th percentile
    threshold_25 = np.percentile(local_variance, 25)
    axes[0, 2].axhline(y=threshold_25, color='r', linestyle='--', 
                       label=f'25th percentile ({threshold_25:.4f})')
    axes[0, 2].legend()
    
    # 4. Smoothest regions highlighted
    colors = ['green' if smooth_mask_viz[i] else 'lightblue' if bbox_mask[i] else 'gray' 
              for i in range(len(y))]
    axes[1, 0].scatter(y, local_variance, alpha=0.6, s=20, c=colors)
    axes[1, 0].set_xlabel('Target Value (normalized)', fontsize=12)
    axes[1, 0].set_ylabel('Local Variance', fontsize=12)
    axes[1, 0].set_title(f'Smoothest {best_n_smooth} Points (green) & Bounding Box (blue)', 
                         fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label=f'{best_n_smooth} smoothest points'),
        Patch(facecolor='lightblue', label=f'Other points in bbox ({best_actual_count - best_n_smooth})'),
        Patch(facecolor='gray', label='Outside bbox')
    ]
    axes[1, 0].legend(handles=legend_elements, loc='upper left')
    
    # 5. Average variance by target value bin
    bin_edges, counts, bin_centers = analyze_target_value_regions(y, n_bins=15)
    avg_variance_per_bin = []
    for i in range(len(bin_centers)):
        bin_mask = (y >= bin_edges[i]) & (y < bin_edges[i+1])
        avg_variance_per_bin.append(local_variance[bin_mask].mean())
    
    axes[1, 1].bar(bin_centers, avg_variance_per_bin, width=(bin_edges[1]-bin_edges[0])*0.8, 
                   alpha=0.7, edgecolor='black')
    axes[1, 1].set_xlabel('Target Value (normalized)', fontsize=12)
    axes[1, 1].set_ylabel('Average Local Variance', fontsize=12)
    axes[1, 1].set_title('Average Local Variance by Target Range', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # 6. Recommendations for target regions
    axes[1, 2].axis('off')
    
    recommendations = "DIMENSION-WISE BOUNDS\n" + "="*40 + "\n\n"
    recommendations += f"Target: ~{target_n_points} smooth points\n"
    recommendations += f"Actual: {best_actual_count} points in region\n"
    recommendations += f"Smoothness: {local_variance[smooth_mask_viz].mean():.4f} avg variance\n\n"
    
    recommendations += "Feature bounds (normalized):\n"
    for i, (feature, (lower, upper)) in enumerate(zip(feature_names, bounds)):
        recommendations += f"{i+1:2d}. {feature:8s}: [{lower:5.2f}, {upper:5.2f}]\n"
        if i == 6:  # Split into two columns
            recommendations += "\n"
    
    recommendations += f"\nTarget value range:\n[{y_smooth.min():.3f}, {y_smooth.max():.3f}]\n"
    
    axes[1, 2].text(0.05, 0.5, recommendations, fontsize=9, family='monospace',
                    verticalalignment='center', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save the figure
    output_path = 'boston_target_region_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()
    
    # Final recommendations
    print("\n" + "="*70)
    print("FINAL RECOMMENDATIONS")
    print("="*70)
    print(f"\nDimension-wise bounds for ~{best_actual_count} smooth points:")
    print(f"Mean local variance: {local_variance[smooth_mask_viz].mean():.4f}")
    print(f"Target value range: [{y_smooth.min():.3f}, {y_smooth.max():.3f}]")
    print("\nCopy the A_limits from the YAML output above to your config file.")
    print("="*70)

if __name__ == "__main__":
    main()
