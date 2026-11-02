import torch
import gpytorch

from .uncertainty import _get_noise_prior, _robust_sigma_w2_seed


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

    including both the kernel removal path K(h) = K - diag(h)K - K diag(h)
    and the target removal path y(h) = y - h ⊙ y -- consistent with
    uncertainty.calculate_aleatoric_uncertainty_influence /
    calculate_aleatoric_hessian (verified in tests/test_curvature_derivatives.py).

    sigma_w^2(h) is tracked via a differentiable Newton refinement of the
    MAP-II stationarity condition (see uncertainty.calculate_aleatoric_hessian),
    seeded by bisection and damped as in uncertainty.calculate_aleatoric_hessian
    (the noise is not refit between removals, so plain Newton could diverge).

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
    Restricted curvature over the submatrix of all current and future removals.

    At removal step k, R_k = T ∩ S^(k) are the samples still present that
    belong to the full removal set T (removed at this step or later).
    H_R(S^(k)) is the principal submatrix of H(S^(k)) = H^epi + H^ale
    restricted to R_k; lambda_min(H_R)/lambda_max(H_R) are the -M_R/-m_R
    bounds of the restricted strong concavity / smoothness condition.

    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs approximating the target region
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
        curvature H[current_index, current_index],
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
    min_k -lambda_max(H_{R_k}) is the restricted strong-concavity constant.
    The first term holds because J >= 0, so Jbar(T) <= J(S).

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
