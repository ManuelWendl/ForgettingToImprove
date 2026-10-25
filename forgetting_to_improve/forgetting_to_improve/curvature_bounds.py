import torch
import gpytorch
from typing import Tuple

from .uncertainty import _get_noise_prior, _robust_sigma_w2_seed


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


def compute_joint_negJ_hessian(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    n_newton_steps: int = 8,
    max_step_fraction: float = 0.5,
    return_value_and_grad: bool = False,
):
    """
    Direct-autodiff Hessian, w.r.t. the RKHS removal directions h, of the
    negated objective

        -J(h) = mean_{x in x_test} F(x,h,sigma_w^2(h)) - sigma_w^2(h),
        F(x,h,s) = k(x,h,S)^T (K(h)+sI)^{-1} k(x,h,S),

    matching Lemma (rkhs-hessian) in proofs.tex, but extended with the
    target removal path y(h) = y - h ⊙ y (not just the kernel path K(h) =
    K - diag(h)K - K diag(h)) -- consistent with
    uncertainty.calculate_aleatoric_uncertainty_influence /
    calculate_aleatoric_hessian, which already include this term.

    This replaces the hand-derived closed form previously assembled from
    compute_epistemic_hessian / compute_aleatoric_hessian / compute_
    aleatoric_hessian_at_x below: those functions do not reproduce their own
    dedicated autodiff checks (see test_epistemic_hessian_autodiff.py /
    test_aleatoric_hessian_autodiff.py at the repo root, which fail against
    them) and do not account for y(h). sigma_w^2(h) is tracked via a
    differentiable Newton refinement of the MAP-II stationarity condition
    (see uncertainty.map_ii_stationarity_residual), initialised at the
    model's current noise level -- which calculate_uncertainty_influence's
    check_stationarity assumes is already (close to) the h=0 root, but which
    optimize.py's removal loops never actually refit between removals. When
    that assumption is stale for the current (reduced) subset, plain Newton
    can diverge (observed as +-inf/nan on real data); see uncertainty.
    calculate_aleatoric_hessian's docstring for the bisection-seeded, damped/
    trust-region-clamped Newton refinement used here to stay finite instead.

    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs used to approximate the integral over the region
        n_newton_steps: Differentiable Newton steps tracking sigma_w^2(h)
        max_step_fraction: Per-step trust-region cap, as a fraction of the
            current s (keeps s within [0.5, 1.5] x its previous value)
        return_value_and_grad: If True, additionally return -J(0) and
            grad_h(-J)(0) (used for the empirical ratio rho_k, see
            calculate_restricted_curvature_bounds)

    Returns:
        H: Hessian of -J w.r.t. h, shape (n_train, n_train), symmetrized
        (if return_value_and_grad) neg_J: scalar -J(0)
        (if return_value_and_grad) grad: grad_h(-J)(0), shape (n_train,)
    """
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(-1)

    kernel = model.covar_module
    n_train = x_samples.shape[0]
    device = x_samples.device

    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    else:
        noise_level = 1e-8

    a, b = 1.0, 0.0
    noise_prior = _get_noise_prior(model.likelihood) if hasattr(model, 'likelihood') else None
    if noise_prior is not None and hasattr(noise_prior, 'concentration'):
        a = noise_prior.concentration.item()
        b = noise_prior.rate.item()

    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()
        K_tx = kernel(x_test, x_samples).evaluate()  # (n_test, n_train)

    I = torch.eye(n_train, dtype=K.dtype, device=device)
    s_seed = _robust_sigma_w2_seed(K, y_samples, a, b, noise_level).detach()

    def neg_J_mean(h: torch.Tensor) -> torch.Tensor:
        y_h = y_samples - h * y_samples
        Dh = torch.diag(h)
        K_h = K - Dh @ K - K @ Dh

        s = s_seed
        for _ in range(n_newton_steps):
            A_h = K_h + s * I
            A_inv_h = torch.linalg.solve(A_h, I)
            alpha1 = A_inv_h @ y_h
            alpha2 = A_inv_h @ alpha1
            g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv_h) + (a - 1.0) / s - b
            A_inv2 = A_inv_h @ A_inv_h
            g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv2) - (a - 1.0) / (s * s)
            g_prime_safe = torch.sign(g_prime + 1e-30) * torch.clamp(g_prime.abs(), min=1e-6)
            step = g / g_prime_safe
            step = torch.clamp(step, min=-max_step_fraction * s, max=max_step_fraction * s)
            s = s - step

        A_h = K_h + s * I
        k_h_all = K_tx - h.unsqueeze(0) * K_tx  # (n_test, n_train)
        sol = torch.linalg.solve(A_h, k_h_all.T)  # (n_train, n_test)
        F_mean = (k_h_all.T * sol).sum(dim=0).mean()
        return F_mean - s

    h0 = torch.zeros(n_train, dtype=K.dtype, device=device, requires_grad=True)
    H = torch.autograd.functional.hessian(neg_J_mean, h0)
    H = 0.5 * (H + H.T)
    if not return_value_and_grad:
        return H
    neg_J = neg_J_mean(h0)
    grad, = torch.autograd.grad(neg_J, h0)
    return H, neg_J.item(), grad.detach()


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
    with 1 at the position of the next sample to remove. H is computed
    directly via autodiff by compute_joint_negJ_hessian (see its docstring
    for why this replaced the previous hand-derived closed form).

    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n, d) - current remaining samples
        y_samples: Training targets (n,)
        x_test: Test points approximating the integration region
        n_integration_samples: Unused (kept for backward-compatible signature)
        next_removal_index: Index of next sample to remove (0 to n-1)
                           If None, returns max eigenvalue (global analysis)

    Returns:
        m: Curvature h^T H h (negative = concave, positive = convex)
           For single index: m = H[i,i] (diagonal element)
    """
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)

    n_train = len(x_samples)

    H = compute_joint_negJ_hessian(model, x_samples, y_samples, x_test)

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


def calculate_restricted_curvature_bounds(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    removal_indices,
    current_index: int = None,
    return_full_grad: bool = False,
):
    """
    Restricted curvature over the *submatrix* of all future removal
    candidates, not just the diagonal entry of the point being removed
    right now.

    This matches the paper's Section M ("Experimental Validation of
    Concavity") / Lemma 5: at removal step k, R_k = T ∩ S^(k) is the set of
    indices (among the samples still present) that belong to the *full*
    eventual removal set T -- i.e. everything that gets removed at this
    step or any later step, not only the single sample chosen right now.
    H_R(S^(k)) is the principal submatrix of H(S^(k)) = H^epi + H^ale
    restricted to R_k, and lambda_min(H_R)/lambda_max(H_R) are exactly the
    -M_R/-m_R spectral bounds of Lemma 5 (Figure 25 in the paper plots
    these two curves directly).

    Args:
        model, x_samples, y_samples, x_test: as in calculate_curvature_bounds
        removal_indices: current-row-position indices (into x_samples) of
            every sample that is a member of T and still present in
            x_samples -- i.e. R_k. Must include current_index.
        current_index: index (into x_samples) of the point actually being
            removed at this step; defaults to removal_indices[0].
        return_full_grad: If True, additionally return the full gradient
            grad_h Jbar(0) = grad_h(-J)(0), shape (n_train,), of the
            removal gain Jbar(T) = J(S) - J(S minus T) -- its top-q entries
            at the full set S give the upper bound U_q (see compute_upper_bound_U).

    Returns:
        (m_diag, lambda_min, lambda_max, J, g): the single-point diagonal
        curvature H[current_index, current_index] (matches the previous
        next_removal_index-only behaviour of calculate_curvature_bounds),
        the min/max eigenvalues of the R_k-restricted submatrix, the
        objective J(S^(k)) and g_k = dJ/dh_i at h=0 (i = current_index), all
        at the MAP-II noise. J and g_k enter the empirical ratio
        rho_k = 2 M_R [J(S^(k)) - J(S^(k+1))] / g_k^2.
    """
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)

    removal_indices = list(removal_indices)
    if current_index is None:
        current_index = int(removal_indices[0])

    H, neg_J, neg_J_grad = compute_joint_negJ_hessian(
        model, x_samples, y_samples, x_test, return_value_and_grad=True
    )

    m_diag = H[current_index, current_index].item()
    J = -neg_J
    g = -neg_J_grad[current_index].item()

    idx = torch.as_tensor(removal_indices, dtype=torch.long, device=H.device)
    H_R = H[idx][:, idx]
    H_R = 0.5 * (H_R + H_R.T)
    eig = torch.linalg.eigvalsh(H_R)

    if return_full_grad:
        return m_diag, eig.min().item(), eig.max().item(), J, g, neg_J_grad
    return m_diag, eig.min().item(), eig.max().item(), J, g


def compute_upper_bound_U(J_S: float, grad_Jbar_0, m_R: float, q_max: int):
    """Upper bound on the best achievable removal gain with q removals,

        Jbar(T*) <= U_q := min{ J(S), sum_{i=1}^q g_(i)^(0) + m_R q / 2 },

    where g_(1)^(0) >= g_(2)^(0) >= ... are the sorted entries of
    grad_h Jbar at the full set S (step 0), Jbar(T) = J(S) - J(S minus T)
    is the removal gain, and m_R = min_k lambda_min(-H_{R_k}) =
    min_k -lambda_max(H_{R_k}) is the restricted strong-concavity constant
    of Lemma 5. The first term holds because J >= 0, so Jbar(T) <= J(S).

    Args:
        J_S: J(S) on the full set, as the (nonnegative) mean predictive
            variance, i.e. including the prior variance k(x,x)
        grad_Jbar_0: grad_h Jbar(0) at the full set, shape (n,)
        m_R: restricted strong-concavity constant
        q_max: largest number of removals q to evaluate

    Returns:
        List [U_0, U_1, ..., U_{q_max}] (U_0 = 0)
    """
    g_sorted = torch.sort(torch.as_tensor(grad_Jbar_0).flatten(), descending=True).values
    top_q_sum = torch.cumsum(g_sorted, dim=0)
    U = [0.0]
    for q in range(1, q_max + 1):
        U.append(min(J_S, top_q_sum[q - 1].item() + m_R * q / 2.0))
    return U


def check_gradient_dominance(grad_inf, J_bar, U_q: float, m_R: float):
    """Gradient-dominance assumption, for every k = 0..q-1:

        ||u^(k) + v^(k)||_inf^2 >= (2 m_R / q) [U_q - Jbar(T_k)]_+,

    with U_q = min{J(S), sum_{i<=q} g_(i)^(0) + m_R q/2} (see
    compute_upper_bound_U), q = len(grad_inf) the number of removals, u^(k) + v^(k) =
    -grad_h Jbar at S^(k) and Jbar(T_k) = J(S) - J(S^(k)).

    Returns:
        (lhs, rhs, met): per-step lists of length q
    """
    q = len(grad_inf)
    lhs = [g ** 2 for g in grad_inf]
    rhs = [2.0 * m_R / q * max(U_q - J_bar[k], 0.0) for k in range(q)]
    met = [l >= r for l, r in zip(lhs, rhs)]
    return lhs, rhs, met


def compute_J(model, x_samples, y_samples, x_test) -> float:
    """Objective J(S) = -(-J)(h=0) for the given sample set."""
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    _, neg_J, _ = compute_joint_negJ_hessian(
        model, x_samples, y_samples, x_test, return_value_and_grad=True
    )
    return -neg_J


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
