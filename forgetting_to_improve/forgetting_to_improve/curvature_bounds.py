import torch
import gpytorch
from typing import Tuple


def calculate_a_i(
    kernel: gpytorch.kernels.Kernel,
    x_samples: torch.Tensor,
    A_limits: Tuple,
    n_integration_samples: int
) -> torch.Tensor:
    n_train = x_samples.shape[0]
    n_dims = x_samples.shape[1]
    device = x_samples.device
    
    # Generate integration samples within A and compute volume
    if n_dims == 1:
        x_A = torch.linspace(A_limits[0], A_limits[1], n_integration_samples, device=device).unsqueeze(-1)
        volume = A_limits[1] - A_limits[0]
    else:
        # Multidimensional case: uniform sampling in hypercube
        if not isinstance(A_limits[0], (list, tuple)):
            A_limits = [A_limits] * n_dims
        
        x_A = torch.rand(n_integration_samples, n_dims, device=device)
        for dim in range(n_dims):
            x_A[:, dim] = x_A[:, dim] * (A_limits[dim][1] - A_limits[dim][0]) + A_limits[dim][0]
        
        volume = 1.0
        for dim_limits in A_limits:
            volume *= (dim_limits[1] - dim_limits[0])
            
    a_i = torch.zeros(n_train, device=device)
    
    with torch.no_grad():
        for i in range(n_train):
            x_i = x_samples[i:i+1]
            # Compute k(x, x_i) for all x in A
            k_values = kernel(x_A, x_i).evaluate().squeeze()
            # Integral: ∫_A k(x, x_i)^2 dx ≈ mean(k²) × volume
            a_i[i] = (k_values ** 2).sum()
    
    return a_i


def calculate_curvature_bounds(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    A_limits: Tuple,
    n_integration_samples: int = 1000
) -> Tuple[float, float]:
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    
    n_train = len(x_samples)
    n_dims = x_samples.shape[1]
    device = x_samples.device
    kernel = model.covar_module
    
    # Get noise level
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    else:
        print("Warning: Model has no likelihood noise attribute; assuming noise level = 1e-8")
        noise_level = 1e-8
    
    # Compute A = K(S, S) + σ_w^2 I
    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()
    A = K + noise_level * torch.eye(n_train, device=device)
    
    # Compute A^{-1}
    try:
        A_inv = torch.inverse(A)
    except Exception as e:
        print(f"Warning: Matrix inversion failed: {e}. Adding jitter.")
        A = A + 1e-4 * torch.eye(n_train, device=device)
        A_inv = torch.inverse(A)
    
    # === Compute quantities for m_epi ===
    
    # Calculate a_i for each training point
    a_i = calculate_a_i(kernel, x_samples, A_limits, n_integration_samples)
    a_min = a_i.min().item()
    
    # Calculate λ_max(A)
    eigenvalues = torch.linalg.eigvalsh(A)
    lambda_max = eigenvalues.max().item()
    
    # Compute m_epi
    m_epi = (2.0 * a_min) / lambda_max
    
    # === Compute quantities for M_ale ===
    
    # Calculate Y = ||y||_2
    Y = torch.norm(y_samples, p=2).item()/len(y_samples)
    
    # Calculate y_max = max_i |y_i|
    y_max = torch.abs(y_samples).max().item()/len(y_samples)
    
    # Calculate q_2 = A^{-2} y
    A_inv_2 = A_inv @ A_inv
    q_2 = A_inv_2 @ y_samples
    
    # Calculate q_3 = A^{-3} y
    A_inv_3 = A_inv_2 @ A_inv
    q_3 = A_inv_3 @ y_samples
    
    # Calculate q_4 = A^{-4} y
    A_inv_4 = A_inv_3 @ A_inv
    q_4 = A_inv_4 @ y_samples
    
    # Calculate d = y^T A^{-3} y - (1/2) tr(A^{-2})
    y_A3_y = torch.dot(y_samples, q_3).item()
    trace_A2 = torch.trace(A_inv_2).item()
    d_value = y_A3_y - 0.5 * trace_A2
    
    # Calculate norms
    norm_q_2 = torch.norm(q_2, p=2).item()
    norm_q_3 = torch.norm(q_3, p=2).item()
    norm_q_4 = torch.norm(q_4, p=2).item()
    
    # Calculate ||A^{-3}||_* (nuclear norm = sum of singular values)
    # For symmetric positive definite matrix, nuclear norm = trace
    nuclear_norm_A_inv_3 = torch.trace(A_inv_3).item()
    
    # Compute volume of region A
    if n_dims == 1:
        volume_A = A_limits[1] - A_limits[0]
    else:
        if not isinstance(A_limits[0], (list, tuple)):
            A_limits_list = [A_limits] * n_dims
        else:
            A_limits_list = A_limits
        volume_A = 1.0
        for dim_limits in A_limits_list:
            volume_A *= (dim_limits[1] - dim_limits[0])
    
    # Compute M_ale = |A| * (term1 + term2 + term3) as per theoretical formula
    term1 = (norm_q_2 ** 2 * y_max ** 2) / d_value
    term2 = (4.0 * y_max ** 2 * norm_q_2 * norm_q_3) / (d_value ** 2)
    term3 = ((3.0 * norm_q_4 * Y + nuclear_norm_A_inv_3) * norm_q_2 ** 2 * y_max ** 2) / (d_value ** 3)
    
    M_ale = volume_A * (term1 + term2 + term3)
    
    return m_epi, M_ale


def compute_curvature_ratio(m_epi: float, M_ale: float) -> float:
    if M_ale == 0:
        return float('inf')
    return m_epi / M_ale
