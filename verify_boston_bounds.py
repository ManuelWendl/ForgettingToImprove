"""Verify how many points fall within the A_limits bounds."""

import numpy as np
import sys
sys.path.append('forgetting_to_improve/forgetting_to_improve')

from helper.data_loader import load_boston

# Load data
x, y = load_boston()

# Current A_limits from boston.yaml (SMALLER REGION - ~33 points)
A_limits = [
    [-0.417, 1.101],   # CRIM
    [-0.488, 2.946],   # ZN
    [-1.133, 1.016],   # INDUS
    [-0.274, -0.272],  # CHAS - FIXED: small tolerance around -0.273
    [-1.354, 0.219],   # NOX
    [-0.822, -0.313],  # RM
    [-2.161, 0.698],   # AGE
    [-0.633, 1.339],   # DIS
    [-0.638, 1.661],   # RAD
    [-1.094, 1.531],   # TAX
    [-1.089, 1.176],   # PTRATIO
    [-0.129, 0.441],   # B
    [-1.000, 0.436],   # LSTAT
]

# Check each dimension
print("Checking each dimension:")
for i, (lower, upper) in enumerate(A_limits):
    mask = (x[:, i] >= lower) & (x[:, i] <= upper)
    n_in_range = mask.sum()
    print(f"  Dim {i+1:2d}: [{lower:7.3f}, {upper:7.3f}] -> {n_in_range:3d} points ({n_in_range/len(x)*100:5.1f}%)")
    
    if n_in_range == 0:
        print(f"    ⚠️  NO POINTS! Data range: [{x[:, i].min():.3f}, {x[:, i].max():.3f}]")

# Check overall
print("\nChecking ALL dimensions together:")
mask = np.ones(len(x), dtype=bool)
for i, (lower, upper) in enumerate(A_limits):
    mask &= (x[:, i] >= lower) & (x[:, i] <= upper)
    n_remaining = mask.sum()
    print(f"  After dim {i+1:2d}: {n_remaining:3d} points remaining")

print(f"\n{'='*60}")
print(f"TOTAL POINTS IN REGION: {mask.sum()}")
print(f"{'='*60}")

# Check CHAS dimension specifically
print("\nCHAS (dimension 4) analysis:")
print(f"  Unique values: {np.unique(x[:, 3])}")
print(f"  Count of each: {np.bincount((x[:, 3] == -0.273).astype(int))}")
print(f"  Closest value to -0.273: {x[:, 3][np.argmin(np.abs(x[:, 3] - (-0.273)))]:10.6f}")
