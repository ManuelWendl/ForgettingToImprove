import torch
import gpytorch
from typing import Tuple, Optional, List, Dict
from itertools import combinations
from .uncertainty import (
    calculate_uncertainty_influence,
    calculate_aleatoric_hessian,
)
from .curvature_bounds import calculate_curvature_bounds
from .helper.plot_gp import visualize_hessian, plot_optimal_comparison


def select_worst_sample(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_type: str = 'joint'
) -> Tuple[Optional[int], torch.Tensor, torch.Tensor]:
    """
    Select the sample to potentially remove.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        noise_type: Type of uncertainty to consider ('joint', 'epistemic', 'aleatoric')
        
    Returns:
        min_index: Index of worst sample (or None if no negative influence)
        epistemic_influence: Epistemic uncertainty influence
        aleatoric_influence: Aleatoric uncertainty influence
    """
    epistemic_influence, aleatoric_influence, factor = (
        calculate_uncertainty_influence(model, x_samples, y_samples, x_test)
    )
    
    if noise_type == 'joint':
        total = epistemic_influence + aleatoric_influence * factor
    elif noise_type == 'epistemic':
        total = epistemic_influence
    elif noise_type == 'aleatoric':
        total = aleatoric_influence
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', or 'aleatoric'.")
    
    min_index = int(total.argmin())
    if total[min_index] < 0:
        return min_index, epistemic_influence, aleatoric_influence
    
    return None, epistemic_influence, aleatoric_influence


def sequentially_optimize_samples(
    model: gpytorch.models.GP,
    likelihood: gpytorch.likelihoods.Likelihood,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    max_iter: int = 20,
    noise_type: str = 'joint',
    show_hessian: bool = False,
    compare_optimal: bool = False,
    calculate_convexity: bool = False,
    A_limits: Tuple = None
) -> Tuple[torch.Tensor, torch.Tensor, List, Dict]:
    """
    Optimize the training samples by removing those that negatively impact the GP model.
    
    Args:
        model: GPyTorch GP model
        likelihood: GPyTorch likelihood
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        max_iter: Maximum iterations
        noise_type: Type of uncertainty to consider
        show_hessian: Whether to visualize Hessian (only for 1D)
        calculate_convexity: Whether to compute curvature bound m
        A_limits: Region limits for curvature bound calculation (required if calculate_convexity=True)
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
        deleted_x_samples: List of removed samples
        curvature_history: Dictionary with 'm' list (empty if calculate_convexity=False)
    """
    deleted_x_samples = []
    deleted_y_samples = []
    curvature_history = {'m': [], 'n_samples': []}
    
    # Store original full dataset for optimal comparison
    if compare_optimal:
        original_x = x_samples.clone()
        original_y = y_samples.clone()
        fti_test_uncertainty = []
        optimal_test_uncertainty = []

    
    # Compute initial curvature bounds if requested
    if calculate_convexity:
        if A_limits is None:
            raise ValueError("A_limits must be provided when calculate_convexity=True")
    
    for iteration in range(max_iter):
        index_to_remove, epistemic, aleatoric = select_worst_sample(
            model, x_samples, y_samples, x_test, noise_type=noise_type
        )
        
        # Compute curvature for the sample about to be removed
        if calculate_convexity and index_to_remove is not None:
            try:
                # Sparse directional curvature H[i,i] for sample i
                m = calculate_curvature_bounds(model, x_samples, y_samples, A_limits,
                                              next_removal_index=index_to_remove)
                curvature_history['m'].append(m)
                curvature_history['n_samples'].append(len(x_samples))
                print(f"Iter {iteration+1}: Removing sample {index_to_remove}, H[{index_to_remove},{index_to_remove}]={m:.6e}, n_samples={len(x_samples)}")
            except Exception as e:
                print(f"Warning: Could not compute curvature bounds at iteration {iteration+1}: {e}")
        
        if show_hessian and x_samples.shape[1] == 1:
            # Only show hessian for 1D case
            kernel = model.covar_module
            noise_level = likelihood.noise.item() if hasattr(likelihood, 'noise') else 1e-8
            with torch.no_grad():
                K = kernel(x_samples, x_samples).evaluate()
            A = K + noise_level * torch.eye(len(x_samples), device=x_samples.device)
            A_inv = torch.inverse(A)
            H = calculate_aleatoric_hessian(A_inv, y_samples)
            visualize_hessian(H, aleatoric, epistemic, index_to_remove, 
                              title=f"Hessian Iteration {iteration + 1}_{noise_type}")
        
        if index_to_remove is None:
            break
        
        deleted_x_samples.append(x_samples[index_to_remove].clone())
        deleted_y_samples.append(y_samples[index_to_remove].clone())
        
        # Remove sample
        mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        mask[index_to_remove] = False
        x_samples = x_samples[mask]
        y_samples = y_samples[mask]
        
        if len(x_samples) <= 1:
            print("Cannot remove more samples (would leave less than 2 samples)")
            break

        if compare_optimal:
            # Store FTI uncertainty after this removal (before refitting)
            # We need to refit the model to the current subset first
            model.set_train_data(x_samples, y_samples, strict=False)
            model.eval()
            likelihood.eval()
            
            with torch.no_grad():
                sigma_x_test = model(x_test).variance.mean().item()
            fti_test_uncertainty.append(sigma_x_test)
            
            # Find optimal subset of current size by exhaustive search
            n_current = len(x_samples)
            n_total = len(original_x)
            
            print(f"\n=== Iteration {iteration + 1}: Finding optimal subset of size {n_current} ===")
            
            best_uncertainty = float('inf')
            
            # Try all possible subsets of size n_current
            n_combinations = 0
            for subset_indices in combinations(range(n_total), n_current):
                n_combinations += 1
                subset_indices = list(subset_indices)
                subset_x = original_x[subset_indices]
                subset_y = original_y[subset_indices]
                
                # Evaluate model with this subset (use current hyperparameters, don't refit)
                model.set_train_data(subset_x, subset_y, strict=False)
                model.eval()
                likelihood.eval()
                
                with torch.no_grad():
                    subset_uncertainty = model(x_test).variance.mean().item()
                
                if subset_uncertainty < best_uncertainty:
                    best_uncertainty = subset_uncertainty
                        
            
            optimal_test_uncertainty.append(best_uncertainty)
            
        # Refit model (just set training data, keep eval mode since we don't reoptimize hyperparameters)
        model.set_train_data(x_samples, y_samples, strict=False)
        model.train()
        likelihood.train()

    # Summary of comparison
    if compare_optimal:
        print("\n=== Final Comparison: FTI vs Optimal Subset Selection ===")
        print(f"{'Subset Size':<15} {'FTI Uncertainty':<20} {'Optimal Uncertainty':<20} {'Difference':<15}")
        print("-" * 70)
        
        # Build list of subset sizes (decreasing)
        n_samples_list = []
        for i in range(len(fti_test_uncertainty)):
            subset_size = len(original_x) - i - 1
            n_samples_list.append(subset_size)
            diff = fti_test_uncertainty[i] - optimal_test_uncertainty[i]
            print(f"{subset_size:<15} {fti_test_uncertainty[i]:<20.6f} {optimal_test_uncertainty[i]:<20.6f} {diff:<15.6f}")
        
        # Create comparison plot
        plot_optimal_comparison(
            fti_uncertainty=fti_test_uncertainty,
            optimal_uncertainty=optimal_test_uncertainty,
            n_samples_list=n_samples_list,
            title="FTI_vs_Optimal_Subset_Selection"
        )
    
    return x_samples, y_samples, deleted_x_samples, curvature_history


def batch_optimize_samples(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_type: str = 'joint'
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batch optimization of training samples.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        noise_type: Type of uncertainty to consider
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
    """
    epistemic, aleatoric, factor = calculate_uncertainty_influence(
        model, x_samples, y_samples, x_test
    )
    
    if noise_type == 'joint':
        total = epistemic + aleatoric * factor
    elif noise_type == 'epistemic':
        total = epistemic
    elif noise_type == 'aleatoric':
        total = aleatoric
    else:
        raise ValueError("Invalid noise_type. Choose from 'joint', 'epistemic', or 'aleatoric'.")
    
    negative_indices = torch.where(total < 0)[0]
    if len(negative_indices) > 0:
        print(f"Removing {len(negative_indices)} samples with negative influence")
        mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        mask[negative_indices] = False
        x_samples = x_samples[mask]
        y_samples = y_samples[mask]
    
    print(f"Final number of samples: {len(x_samples)}")
    return x_samples, y_samples