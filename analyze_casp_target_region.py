"""
Analyze CASP dataset to identify good target regions.

This script loads the CASP dataset and analyzes it to find regions
with relatively smooth behavior and low noise for defining target regions.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import sys
sys.path.append('forgetting_to_improve/forgetting_to_improve')

from helper.data_loader import load_casp

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
    print("Loading CASP dataset...")
    x, y = load_casp()
    
    print(f"\nDataset statistics:")
    print(f"  Number of samples: {len(y)}")
    print(f"  Number of features: {x.shape[1]}")
    print(f"  Target (y) range: [{y.min():.3f}, {y.max():.3f}]")
    print(f"  Target (y) mean: {y.mean():.3f}, std: {y.std():.3f}")
    
    # Show feature statistics
    print(f"\nFeature statistics (normalized):")
    for i in range(x.shape[1]):
        print(f"  F{i+1}: min={x[:, i].min():.3f}, max={x[:, i].max():.3f}, "
              f"mean={x[:, i].mean():.3f}, std={x[:, i].std():.3f}")
    
    # Analyze local smoothness
    print("\nAnalyzing local smoothness...")
    local_variance = analyze_local_smoothness(x, y, n_neighbors=10)
    
    print(f"  Local variance range: [{local_variance.min():.4f}, {local_variance.max():.4f}]")
    print(f"  Local variance mean: {local_variance.mean():.4f}, std: {local_variance.std():.4f}")
    
    # Target: Find region with approximately 20-100 smooth points
    # Try different approaches to find a region that's not too small and not too large
    
    print(f"\n{'='*80}")
    print(f"Finding smooth regions with various strategies:")
    print(f"{'='*80}")
    
    best_config = None
    best_score = float('inf')
    
    # Strategy 1: Use smoothest points with expanding margins
    n_smooth_base = 10
    smooth_indices = np.argsort(local_variance)[:n_smooth_base]
    x_smooth = x[smooth_indices]
    y_smooth = y[smooth_indices]
    
    # Try different margin sizes to expand the region
    margin_factors = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    for margin_factor in margin_factors:
        bounds = []
        for i in range(x.shape[1]):
            dim_min = x_smooth[:, i].min()
            dim_max = x_smooth[:, i].max()
            dim_range = dim_max - dim_min
            
            # Add margin proportional to the range
            margin = margin_factor * dim_range if dim_range > 0.01 else 0.01
            bounds.append([dim_min - margin, dim_max + margin])
        
        # Count points in these bounds
        n_in_bounds = count_points_in_bounds(x, bounds)
        
        # Check local variance in region
        mask = np.ones(len(x), dtype=bool)
        for i, (lower, upper) in enumerate(bounds):
            mask &= (x[:, i] >= lower) & (x[:, i] <= upper)
        
        if mask.sum() >= 10:
            local_var_in_region = local_variance[mask]
            avg_local_var = local_var_in_region.mean()
            
            # Score: prefer 20-100 points with low variance
            # Penalize regions that are too small (<20) or too large (>100)
            size_penalty = 0
            if n_in_bounds < 20:
                size_penalty = (20 - n_in_bounds) * 0.01
            elif n_in_bounds > 100:
                size_penalty = (n_in_bounds - 100) * 0.001
            
            score = avg_local_var + size_penalty
            
            print(f"\nMargin factor {margin_factor:.2f}:")
            print(f"  Points in bounds: {n_in_bounds}")
            print(f"  Avg local variance: {avg_local_var:.4f}")
            print(f"  Score: {score:.4f}")
            print(f"  Bounds:")
            for i, (lower, upper) in enumerate(bounds):
                print(f"    F{i+1}: [{lower:.3f}, {upper:.3f}]  (width: {upper-lower:.3f})")
            
            # Track best configuration (lowest score)
            if score < best_score and 20 <= n_in_bounds <= 100:
                best_score = score
                best_config = {
                    'n_smooth': n_smooth_base,
                    'margin_factor': margin_factor,
                    'n_in_bounds': n_in_bounds,
                    'bounds': bounds,
                    'smooth_indices': smooth_indices,
                    'avg_local_var': avg_local_var
                }
    
    if best_config is not None:
        print(f"\n{'='*80}")
        print(f"RECOMMENDED CONFIGURATION (best smooth region):")
        print(f"{'='*80}")
        print(f"Smooth points (seed): {best_config['n_smooth']}")
        print(f"Margin factor: {best_config['margin_factor']:.2f}")
        print(f"Points in bounds: {best_config['n_in_bounds']}")
        print(f"Avg local variance: {best_config['avg_local_var']:.4f}")
        print(f"\nA_limits for casp.yaml:")
        print("  A_limits: [")
        for i, (lower, upper) in enumerate(best_config['bounds']):
            print(f"    [{lower:.3f}, {upper:.3f}],  # F{i+1}")
        print("  ]")
        
        # Verify that all bounds have reasonable width
        print(f"\nBound widths:")
        for i, (lower, upper) in enumerate(best_config['bounds']):
            width = upper - lower
            print(f"  F{i+1}: {width:.4f}")
        
        # Show which points are in the region
        mask = np.ones(len(x), dtype=bool)
        for i, (lower, upper) in enumerate(best_config['bounds']):
            mask &= (x[:, i] >= lower) & (x[:, i] <= upper)
        
        if mask.sum() > 0:
            x_in_region = x[mask]
            y_in_region = y[mask]
            print(f"\nPoints in target region:")
            print(f"  Count: {len(y_in_region)}")
            print(f"  Target (y) range: [{y_in_region.min():.3f}, {y_in_region.max():.3f}]")
            print(f"  Target (y) mean: {y_in_region.mean():.3f}, std: {y_in_region.std():.3f}")
            
            # Check local variance in region
            if len(y_in_region) >= 5:
                local_var_in_region = local_variance[mask]
                print(f"  Local variance in region:")
                print(f"    Mean: {local_var_in_region.mean():.4f}")
                print(f"    Std: {local_var_in_region.std():.4f}")
                print(f"    Max: {local_var_in_region.max():.4f}")
                print(f"  Compared to dataset average: {local_variance.mean():.4f}")
    else:
        print("\nNo suitable configuration found. Consider adjusting parameters.")

if __name__ == '__main__':
    main()
