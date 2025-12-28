from typing import Tuple
import torch
import gpytorch
from math import comb
from itertools import combinations


def calculate_epistemic_uncertainty_influence(
    A_inv: torch.Tensor, 
    model: gpytorch.models.GP, 
    x_samples: torch.Tensor, 
    x_test: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the epistemic uncertainty of the GP predictions at points X.
    
    Args:
        A_inv: Inverse of the covariance matrix (n_train, n_train)
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        x_test: Test inputs (n_test, d)
        
    Returns:
        epistemic_influence: Epistemic uncertainty influence (n_train,)
        epistemic_uncertainty_marginal: Marginal epistemic uncertainty (n_train,)
    """
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(-1)
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    
    kernel = model.covar_module
    
    with torch.no_grad():
        K_tx = kernel(x_test, x_samples).evaluate()
    
    epistemic_influence = 2 * torch.sum(K_tx.T * (A_inv @ K_tx.T), dim=1)
    
    epistemic_uncertainty_marginal = (
        1 / torch.diag(A_inv) * torch.sum((A_inv @ K_tx.T) ** 2, dim=1)
    )
    
    return epistemic_influence, epistemic_uncertainty_marginal


def calculate_epistemic_shapley(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_level: float = 1e-8
) -> torch.Tensor:
    """
    Calculate the exact Shapley values for epistemic uncertainty.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        x_test: Test inputs (n_test, d)
        noise_level: Noise level for regularization
        
    Returns:
        shapley_values: Shapley values (n_test, n_train)
    """
    # Ensure proper shape
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(-1)
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    
    n_train = x_samples.shape[0]
    n_test = x_test.shape[0]
    kernel = model.covar_module
    
    # Get noise level from likelihood if available
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    
    # Initialize Shapley values
    shapley_values = torch.zeros((n_test, n_train), device=x_samples.device)
    
    def epistemic_uncertainty_subset(subset_indices):
        """Calculate epistemic uncertainty for a subset of training points."""
        if len(subset_indices) == 0:
            # No training data: uncertainty equals prior variance
            with torch.no_grad():
                K_xx = kernel(x_test, x_test).evaluate()
                return torch.diag(K_xx)
        
        # Get subset of training points
        x_subset = x_samples[subset_indices]
        
        # Compute kernel matrices for subset
        with torch.no_grad():
            K_subset = kernel(x_subset, x_subset).evaluate()
        
        # Add noise term
        A_subset = K_subset + noise_level * torch.eye(
            len(subset_indices), device=x_samples.device
        )
        
        try:
            A_subset_inv = torch.inverse(A_subset)
        except RuntimeError:
            # Add regularization if singular
            A_subset = K_subset + (noise_level + 1e-6) * torch.eye(
                len(subset_indices), device=x_samples.device
            )
            A_subset_inv = torch.inverse(A_subset)
        
        # Cross-covariance between test and subset
        with torch.no_grad():
            K_tx_subset = kernel(x_test, x_subset).evaluate()
            K_xx = kernel(x_test, x_test).evaluate()
        
        # Epistemic uncertainty: K(x*,x*) - K(x*,X_S) A_S^{-1} K(X_S,x*)
        epistemic_var = torch.diag(K_xx) - torch.sum(
            K_tx_subset * (A_subset_inv @ K_tx_subset.T).T, dim=1
        )
        
        return epistemic_var
    
    # Calculate Shapley values for each training point
    for i in range(n_train):
        shapley_value_i = torch.zeros(n_test, device=x_samples.device)
        
        # Sum over all subsets S ⊆ N\{i}
        other_indices = [j for j in range(n_train) if j != i]
        
        for subset_size in range(n_train):  # |S| from 0 to n-1
            # Weight for this subset size
            weight = 1.0 / (n_train * comb(n_train - 1, subset_size))
            
            # Iterate over all subsets of given size
            for subset in combinations(other_indices, subset_size):
                subset_list = list(subset)
                
                # f(S ∪ {i}) - f(S)
                uncertainty_with_i = epistemic_uncertainty_subset(subset_list + [i])
                uncertainty_without_i = epistemic_uncertainty_subset(subset_list)
                
                marginal_contribution = uncertainty_without_i - uncertainty_with_i
                shapley_value_i += weight * marginal_contribution
        
        shapley_values[:, i] = shapley_value_i
    
    return shapley_values


def calculate_aleatoric_uncertainty_influence(
    A_inv: torch.Tensor, 
    y_samples: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the aleatoric uncertainty of the GP predictions at points X.
    
    Args:
        A_inv: Inverse of the covariance matrix (n_train, n_train)
        y_samples: Training targets (n_train,)
        
    Returns:
        aleatoric_influence: Aleatoric uncertainty influence (n_train,)
    """
    alpha = A_inv @ y_samples
    Am2y = A_inv @ (A_inv @ y_samples)      # = A^{-2} y
    Am3y = torch.dot(alpha, A_inv @ alpha)        # = y^T A^{-3} y
    trAm2 = torch.trace(A_inv @ A_inv)
    H = -Am3y + 0.5 * trAm2
    return Am2y / H * y_samples


def calculate_aleatoric_hessian(
    A_inv: torch.Tensor, 
    y_samples: torch.Tensor
) -> torch.Tensor:
    """
    Calculate the aleatoric uncertainty Hessian.
    
    Args:
        A_inv: Inverse of the covariance matrix (n_train, n_train)
        y_samples: Training targets (n_train,)
        
    Returns:
        H: Hessian matrix (n_train, n_train)
    """
    alpha = A_inv @ y_samples
    N = A_inv @ (A_inv @ y_samples)      # = A^{-2} y
    Am3y = torch.dot(alpha, A_inv @ alpha)        # = y^T A^{-3} y
    R = A_inv @ N       # = A^{-3} y
    trAm2 = torch.trace(A_inv @ A_inv)
    D = -Am3y + 0.5 * trAm2
    v = -N / D
    S = 3 * torch.dot(y_samples, A_inv @ R) - torch.trace(A_inv @ A_inv @ A_inv)
    
    # Use outer products to ensure symmetry
    H = (
        -1 / D * (A_inv @ A_inv - torch.outer(R, v) - torch.outer(v, R))
        + 1 / D**2 * (
            torch.outer(N, -2 * R + S * v) + torch.outer(-2 * R + S * v, N)
        )
    )
    return H


def calculate_uncertainty_influence(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the uncertainty (epistemic and aleatoric) of the GP predictions at points X.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        
    Returns:
        epistemic_influence: Epistemic uncertainty influence (n_train,)
        aleatoric_influence: Aleatoric uncertainty influence (n_train,)
        epistemic_uncertainty_marginal: Marginal epistemic uncertainty (n_train,)
    """
    # Ensure proper shape
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    
    # Get kernel and noise level
    kernel = model.covar_module
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    else:
        noise_level = 1e-8
    
    # Compute covariance matrix and its inverse
    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()
    
    A = K + noise_level * torch.eye(len(x_samples), device=x_samples.device)
    A_inv = torch.inverse(A)
    
    epistemic_influence, epistemic_uncertainty_marginal = (
        calculate_epistemic_uncertainty_influence(A_inv, model, x_samples, x_test)
    )
    aleatoric_influence = calculate_aleatoric_uncertainty_influence(A_inv, y_samples)
    
    return epistemic_influence, aleatoric_influence, epistemic_uncertainty_marginal