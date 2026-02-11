import torch
import gpytorch
from typing import Tuple


def compute_aleatoric_hessian(
    A_inv: torch.Tensor,
    y_samples: torch.Tensor,
    K: torch.Tensor,
    noise_level: float,
    gamma_prior_a: float = 2.0,
    gamma_prior_b: float = 0.15
) -> torch.Tensor:
    """
    Compute the aleatoric Hessian ∇²_h σ²_w using the implicit function theorem.
    
    This follows Lemma (aleatoric-hessian) from the paper.
    
    Args:
        A_inv: A^{-1} where A = K + σ²_w I
        y_samples: Training targets (n,)
        K: Covariance matrix K(S, S)
        noise_level: Current estimate of σ²_w
        gamma_prior_a, gamma_prior_b: Gamma prior parameters
        
    Returns:
        Hessian matrix ∇²_h σ²_w of shape (n, n)
    """
    n_train = len(y_samples)
    device = y_samples.device
    
    # Compute powers of A^{-p}
    A_inv_2 = A_inv @ A_inv
    A_inv_3 = A_inv_2 @ A_inv
    A_inv_4 = A_inv_3 @ A_inv
    
    # Compute α^(p) = A^{-p}y
    alpha_1 = A_inv @ y_samples
    alpha_2 = A_inv_2 @ y_samples
    alpha_3 = A_inv_3 @ y_samples
    
    # Compute M^(p) = KA^{-p}
    M_1 = K @ A_inv
    M_2 = K @ A_inv_2
    M_3 = K @ A_inv_3
    
    # Compute s^(p) = Kα^(p)
    s_1 = K @ alpha_1
    s_2 = K @ alpha_2
    s_3 = K @ alpha_3
    
    # Compute D (denominator)
    D = -torch.dot(alpha_1, alpha_2) + 0.5 * torch.trace(A_inv_2) - (gamma_prior_a - 1) / (noise_level ** 2)
    
    # Compute gradient ∇_h σ²_w
    grad_sigma = (1.0 / D) * (-alpha_1 * s_2 - alpha_2 * s_1 + torch.diag(M_2))
    
    # Compute mixed derivative ∇_{σ²_w,h}² L
    mixed_deriv = 2 * alpha_2 * s_2 + 2 * alpha_1 * s_3 + 2 * alpha_3 * s_1 - 2 * torch.diag(M_3)
    
    # Compute ∇²_h L (the complex term)
    # Breaking it down into parts for clarity
    term1 = A_inv * (s_2.unsqueeze(1) @ alpha_1.unsqueeze(0) + alpha_1.unsqueeze(1) @ s_2.unsqueeze(0))
    term2_asym = A_inv_2 * (s_1.unsqueeze(1) @ alpha_1.unsqueeze(0))
    term2 = term2_asym + term2_asym.T  # Symmetrize
    term3 = torch.diag(s_2) @ M_1 @ A_inv
    term4 = torch.diag(s_1) @ M_1 @ A_inv_2
    term5 = term3.T + term4.T
    term6 = -M_1 * M_2.T - M_2 * M_1.T
    
    hess_L = term1 + term2 + term3 + term4 + term5 + term6
    
    # Compute ∇_{σ²_w} D
    grad_D_sigma = 3 * torch.dot(alpha_2, alpha_2) - torch.trace(A_inv_3) + 2 * (gamma_prior_a - 1) / (noise_level ** 3)
    
    # Assemble final Hessian
    hess_sigma = -(1.0 / D) * (
        hess_L +
        mixed_deriv.unsqueeze(1) @ grad_sigma.unsqueeze(0) +
        grad_sigma.unsqueeze(1) @ mixed_deriv.unsqueeze(0) +
        grad_D_sigma * (grad_sigma.unsqueeze(1) @ grad_sigma.unsqueeze(0))
    )
    
    return hess_sigma


def compute_epistemic_hessian(
    A_inv: torch.Tensor,
    K: torch.Tensor,
    x_test: torch.Tensor,
    kernel: gpytorch.kernels.Kernel,
    x_samples: torch.Tensor
) -> torch.Tensor:
    """
    Compute the epistemic Hessian H^epi(x) at test point x.
    
    This follows Lemma (rkhs-hessian) from the paper.
    
    Args:
        A_inv: A^{-1} where A = K + σ²_w I
        K: Covariance matrix K(S, S)
        x_test: Single test point (1, d)
        kernel: Kernel function
        x_samples: Training inputs (n, d)
        
    Returns:
        Epistemic Hessian H^epi(x) of shape (n, n)
    """
    # Compute c_x = A^{-1} k(S, x)
    with torch.no_grad():
        k_Sx = kernel(x_samples, x_test).evaluate().squeeze()
    c_x = A_inv @ k_Sx
    
    # Compute M^(1) = KA^{-1} and M^(2) = A^{-1}KA^{-1}
    M_1 = K @ A_inv
    M_2 = A_inv @ K @ A_inv
    
    # Compute H^epi(x) = 2(diag(c_x)(M^(2) + M^(2)ᵀ)diag(c_x) - 2c_x c_xᵀ ⊙ M^(1) - M^(1) ⊙ c_x c_xᵀ)
    diag_cx = torch.diag(c_x)
    cx_cxT = c_x.unsqueeze(1) @ c_x.unsqueeze(0)
    
    H_epi = 2 * (
        diag_cx @ (M_2 + M_2.T) @ diag_cx
        - 2 * cx_cxT * M_1
        - M_1 * cx_cxT
    )
    
    return H_epi


def compute_aleatoric_hessian_at_x(
    A_inv: torch.Tensor,
    K: torch.Tensor,
    x_test: torch.Tensor,
    kernel: gpytorch.kernels.Kernel,
    x_samples: torch.Tensor,
    grad_sigma: torch.Tensor,
    hess_sigma: torch.Tensor
) -> torch.Tensor:
    """
    Compute the aleatoric Hessian H^ale(x) at test point x.
    
    This follows Lemma (rkhs-hessian) from the paper.
    
    Args:
        A_inv: A^{-1} where A = K + σ²_w I
        K: Covariance matrix K(S, S)
        x_test: Single test point (1, d)
        kernel: Kernel function
        x_samples: Training inputs (n, d)
        grad_sigma: Gradient ∇_h σ²_w from aleatoric_hessian computation
        hess_sigma: Hessian ∇²_h σ²_w from aleatoric_hessian computation
        
    Returns:
        Aleatoric Hessian H^ale(x) of shape (n, n)
    """
    # Compute c_x = A^{-1} k(S, x)
    with torch.no_grad():
        k_Sx = kernel(x_samples, x_test).evaluate().squeeze()
    c_x = A_inv @ k_Sx
    
    # Compute M^(1) = KA^{-1}
    M_1 = K @ A_inv
    
    # Compute ||c_x||²
    norm_cx_sq = torch.dot(c_x, c_x)
    
    # Compute ∇_h ||c_x||² = 2 diag(c_x)(M^(1)c_x - c_x ⊙ diag(M^(1)))
    M1_cx = M_1 @ c_x
    diag_M1 = torch.diag(M_1)
    grad_norm_cx = 2 * torch.diag(c_x) @ (M1_cx - c_x * diag_M1)
    
    # Compute H^ale(x) with symmetrization
    # The outer product needs to be symmetric: v⊗w + w⊗v
    outer = grad_norm_cx.unsqueeze(1) @ grad_sigma.unsqueeze(0)
    H_ale = -(outer + outer.T) - (norm_cx_sq + 1) * hess_sigma
    
    return H_ale


def integrate_hessians_over_region(
    A_inv: torch.Tensor,
    K: torch.Tensor,
    kernel: gpytorch.kernels.Kernel,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    noise_level: float,
    x_test: Tuple,
    n_integration_samples: int = 100,
    gamma_prior_a: float = 2,
    gamma_prior_b: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate the epistemic and aleatoric Hessians over region A.
    
    Returns:
        H_epi_integrated: ∫_A H^epi(x) dx
        H_ale_integrated: ∫_A H^ale(x) dx
    """
    n_train = x_samples.shape[0]
    device = x_samples.device
    
    # First compute the aleatoric Hessian components (independent of x)
    hess_sigma = compute_aleatoric_hessian(
        A_inv, y_samples, K, noise_level, gamma_prior_a, gamma_prior_b
    )
    
    # Compute gradient for aleatoric term
    A_inv_2 = A_inv @ A_inv
    alpha_1 = A_inv @ y_samples
    alpha_2 = A_inv_2 @ y_samples
    M_2 = K @ A_inv_2
    s_1 = K @ alpha_1
    s_2 = K @ alpha_2
    D = -torch.dot(alpha_1, alpha_2) + 0.5 * torch.trace(A_inv_2) - (gamma_prior_a - 1) / (noise_level ** 2)
    grad_sigma = (1.0 / D) * (-alpha_1 * s_2 - alpha_2 * s_1 + torch.diag(M_2))
    
    # Integrate Hessians over test points
    H_epi_sum = torch.zeros((n_train, n_train), device=device)
    H_ale_sum = torch.zeros((n_train, n_train), device=device)
    
    for i in range(len(x_test)):
        x_t = x_test[i:i+1]
        
        # Compute Hessians at this test point
        H_epi_x = compute_epistemic_hessian(A_inv, K, x_t, kernel, x_samples)
        H_ale_x = compute_aleatoric_hessian_at_x(A_inv, K, x_t, kernel, x_samples, grad_sigma, hess_sigma)
        
        H_epi_sum += H_epi_x
        H_ale_sum += H_ale_x
    
    # Scale by volume / n_samples to approximate integral
    H_epi_integrated = (1 / len(x_test)) * H_epi_sum
    H_ale_integrated = (1 / len(x_test)) * H_ale_sum
    
    # The formulas compute ∇²σ², but the objective is J̄ = -∫σ²
    # Therefore: ∇²J̄ = -∫∇²σ² = -∫[H^epi + H^ale]
    return (H_epi_integrated + H_ale_integrated)


def calculate_curvature_bounds(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: Tuple,
    n_integration_samples: int = 1000,
    next_removal_index: int = None
) -> float:
    """
    Calculate curvature bounds using sparse directional derivative.
    
    Computes h^T H h where H is the Hessian ∇²J̄ and h is a sparse vector
    with 1 at the position of the next sample to remove.
    
    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n, d) - current remaining samples
        y_samples: Training targets (n,)
        A_limits: Region limits for integration
        n_integration_samples: Number of samples for numerical integration
        next_removal_index: Index of next sample to remove (0 to n-1)
                           If None, returns max eigenvalue (global analysis)
        
    Returns:
        m: Curvature h^T H h (negative = concave, positive = convex)
           For single index: m = H[i,i] (diagonal element)
    """
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    
    n_train = len(x_samples)
    device = x_samples.device
    kernel = model.covar_module
    
    # Get noise level
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    else:
        print("Warning: Model has no likelihood noise attribute; assuming noise level = 1e-8")
        noise_level = 1e-8
    
    # Compute A = K(S, S) + σ²_w I
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
    
    # Integrate Hessians over region A
    H = integrate_hessians_over_region(
        A_inv=A_inv,
        K=K,
        kernel=kernel,
        x_samples=x_samples,
        y_samples=y_samples,
        noise_level=noise_level,
        x_test=x_test,
        n_integration_samples=n_integration_samples
    )
    
    # Compute sparse directional curvature h^T H h
    if next_removal_index is not None:
        # For sparse direction h = [0,...,0,1,0,...,0] at position i
        # h^T H h = H[i,i] (diagonal element)
        # This corresponds to the curvature when removing x_samples[i]
        if next_removal_index < 0 or next_removal_index >= n_train:
            raise ValueError(f"next_removal_index {next_removal_index} out of range [0, {n_train-1}]")
        m = H[next_removal_index, next_removal_index].item()
        # Verify: sample coordinates for clarity
        if next_removal_index < n_train:
            sample_coords = x_samples[next_removal_index].cpu().numpy() if hasattr(x_samples, 'cpu') else x_samples[next_removal_index]
            print(f"  Sample {next_removal_index} at x={sample_coords}, curvature H[{next_removal_index},{next_removal_index}]={m:.6e}")
    else:
        # Global analysis: compute maximum eigenvalue
        try:
            eig = torch.linalg.eigvalsh(H)
            m = eig.max().item()
        except Exception as e:
            print(f"Warning: Eigenvalue computation failed: {e}")
            m = torch.trace(H).item() / n_train
    
    return m


def compute_curvature_ratio(m_epi: float, M_ale: float) -> float:
    """
    Compute the curvature ratio m_epi / M_ale.
    
    For the objective to be concave:
    - We want m_epi < 0 (epistemic contribution is concave)
    - We want M_ale to be small (aleatoric doesn't destroy concavity)
    - A large negative ratio |m_epi| / M_ale indicates strong concavity
    
    Args:
        m_epi: Minimum eigenvalue of epistemic Hessian
        M_ale: Maximum absolute eigenvalue of aleatoric Hessian
        
    Returns:
        Ratio m_epi / M_ale
    """
    if M_ale == 0:
        return float('-inf') if m_epi < 0 else float('inf')
    return m_epi / M_ale
