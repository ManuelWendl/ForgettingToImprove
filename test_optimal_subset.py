"""
Test script for optimal subset selection via exhaustive search.
This will compare the Forgetting to Improve greedy approach with exhaustive search.
"""

import torch
import gpytorch
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath('forgetting_to_improve'))

from forgetting_to_improve.models import ExactGPModel
from forgetting_to_improve.optimize import sequentially_optimize_samples

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate simple 1D synthetic data
n_train = 10
n_test = 50

# Training data
x_train = torch.linspace(0, 1, n_train).unsqueeze(-1)
y_train = torch.sin(2 * np.pi * x_train.squeeze()) + 0.1 * torch.randn(n_train)

# Test data
x_test = torch.linspace(0, 1, n_test).unsqueeze(-1)

print("=" * 70)
print("Testing Optimal Subset Selection via Exhaustive Search")
print("=" * 70)
print(f"\nInitial training set size: {n_train}")
print(f"Test set size: {n_test}")

# Initialize model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_train, y_train, likelihood, kernel_type='matern', nu=2.5)

# Train initial model
model.train()
likelihood.train()

print("\n" + "=" * 70)
print("Running FTI with optimal comparison...")
print("=" * 70)

# Run optimization with compare_optimal=True
x_optimized, y_optimized, deleted_samples, _ = sequentially_optimize_samples(
    model=model,
    likelihood=likelihood,
    x_samples=x_train,
    y_samples=y_train,
    x_test=x_test,
    max_iter=3,  # Limit iterations for feasibility (C(10,7)=120, C(10,6)=210, C(10,5)=252)
    noise_type='joint',
    compare_optimal=True
)

print("\n" + "=" * 70)
print("Optimization Complete!")
print("=" * 70)
print(f"\nFinal optimized set size: {len(x_optimized)}")
print(f"Number of samples removed: {len(deleted_samples)}")
