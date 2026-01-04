import torch
import gpytorch
from typing import Tuple, Optional
from .uncertainty import (
    calculate_uncertainty_influence,
    calculate_aleatoric_hessian,
)
from .helper.plot_gp import visualize_hessian


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
    epistemic_influence, aleatoric_influence, epistemic_uncertainty_marginal = (
        calculate_uncertainty_influence(model, x_samples, y_samples, x_test)
    )
    
    if noise_type == 'joint':
        total = epistemic_influence + aleatoric_influence
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
    show_hessian: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, list]:
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
        
    Returns:
        x_samples: Optimized training inputs
        y_samples: Optimized training targets
        deleted_x_samples: List of removed samples
    """
    deleted_x_samples = []
    
    for iteration in range(max_iter):
        index_to_remove, epistemic, aleatoric = select_worst_sample(
            model, x_samples, y_samples, x_test, noise_type=noise_type
        )
        
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
        
        # Remove sample
        mask = torch.ones(len(x_samples), dtype=torch.bool, device=x_samples.device)
        mask[index_to_remove] = False
        x_samples = x_samples[mask]
        y_samples = y_samples[mask]
        
        if len(x_samples) <= 1:
            print("Cannot remove more samples (would leave less than 2 samples)")
            break
        
        # Refit model
        model.set_train_data(x_samples, y_samples, strict=False)
        model.train()
        likelihood.train()
    
    return x_samples, y_samples, deleted_x_samples


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
    epistemic, aleatoric, _ = calculate_uncertainty_influence(
        model, x_samples, y_samples, x_test
    )
    
    if noise_type == 'joint':
        total = epistemic + aleatoric
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