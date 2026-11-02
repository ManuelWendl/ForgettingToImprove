"""Autodiff / finite-difference verification of uncertainty.py's Hessians.

Checks two things for both the epistemic and aleatoric Hessian w.r.t. the
RKHS removal directions h:

1. Gradient consistency: torch.autograd.grad of the exact scalar function used
   inside calculate_epistemic_hessian / calculate_aleatoric_hessian (evaluated
   at h=0) reproduces the independently hand-derived closed-form gradients
   calculate_epistemic_uncertainty_influence / calculate_aleatoric_uncertainty_influence.
2. Finite-difference Hessian: central differences of the closed-form gradient
   w.r.t. h (an independent numerical estimate, not reusing torch.autograd.functional
   .hessian internals) match calculate_epistemic_hessian / calculate_aleatoric_hessian.

Removal path used throughout (see uncertainty.py's docstrings):
    k(x,h,S) = k(x,S) - h ⊙ k(x,S)
    K(h)     = K - diag(h)K - K diag(h)   (linearized training-training block)
    y(h)     = y - h ⊙ y
"""
import torch
import gpytorch

from forgetting_to_improve.forgetting_to_improve.uncertainty import (
    calculate_epistemic_uncertainty_influence,
    calculate_epistemic_hessian,
    calculate_aleatoric_uncertainty_influence,
    calculate_aleatoric_hessian,
    _get_noise_prior,
)
from forgetting_to_improve.forgetting_to_improve.curvature_bounds import (
    compute_joint_negJ_hessian,
)


def _make_problem(n=6, d=2, seed=0, noise=0.4, gamma_a=2.0, gamma_b=0.5):
    torch.manual_seed(seed)
    dtype = torch.float64

    x_samples = torch.randn(n, d, dtype=dtype)
    y_samples = torch.randn(n, dtype=dtype)
    x_test = torch.randn(4, d, dtype=dtype)

    kernel = gpytorch.kernels.RBFKernel().to(dtype=dtype)
    kernel.lengthscale = 1.2

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=gpytorch.priors.GammaPrior(gamma_a, gamma_b)
    ).to(dtype=dtype)
    likelihood.noise = noise

    class M:
        covar_module = kernel

    m = M()
    m.likelihood = likelihood

    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()
    A = K + likelihood.noise.item() * torch.eye(n, dtype=dtype)
    A_inv = torch.inverse(A)

    return m, x_samples, y_samples, x_test, K, A_inv


def _solve_map_noise(K, y, a, b, s_init, n_steps=60):
    n = y.numel()
    I = torch.eye(n, dtype=K.dtype)
    s = s_init
    for _ in range(n_steps):
        A = K + s * I
        A_inv = torch.linalg.solve(A, I)
        alpha1 = A_inv @ y
        alpha2 = A_inv @ alpha1
        g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv) + (a - 1.0) / s - b
        A_inv2 = A_inv @ A_inv
        g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv2) - (a - 1.0) / (s * s)
        s = s - g / g_prime
    return s


def test_epistemic_gradient_matches_influence_function():
    model, x_samples, y_samples, x_test, K, A_inv = _make_problem()
    n = x_samples.shape[0]
    noise_level = model.likelihood.noise.item()

    epi_influence, _ = calculate_epistemic_uncertainty_influence(A_inv, model, x_samples, x_test)

    with torch.no_grad():
        K_tx = model.covar_module(x_test, x_samples).evaluate()

    I = torch.eye(n, dtype=K.dtype)
    grad_sum = torch.zeros(n, dtype=K.dtype)
    for i in range(x_test.shape[0]):
        h = torch.zeros(n, dtype=K.dtype, requires_grad=True)
        Dh = torch.diag(h)
        K_h = K - Dh @ K - K @ Dh
        A_h = K_h + noise_level * I
        k_h = K_tx[i] - h * K_tx[i]
        F = k_h @ torch.linalg.solve(A_h, k_h)
        g, = torch.autograd.grad(F, h)
        grad_sum += -g  # sigma_epi^2 = k(x,x) - F

    grad_mean = grad_sum / x_test.shape[0]

    err = (grad_mean - epi_influence).abs().max().item()
    print("epistemic gradient max abs err:", err)
    assert err < 1e-9, f"epistemic influence disagrees with autodiff gradient: {err:.3e}"


def test_aleatoric_gradient_matches_influence_function():
    model, x_samples, y_samples, x_test, K, A_inv = _make_problem()
    n = x_samples.shape[0]

    a = _get_noise_prior(model.likelihood).concentration.item()
    b = _get_noise_prior(model.likelihood).rate.item()
    noise_level = model.likelihood.noise.item()

    # Model noise must actually be the MAP-II root for the implicit-function
    # gradient to be meaningful -- re-solve and overwrite it.
    s0 = _solve_map_noise(K, y_samples, a, b, torch.tensor(noise_level, dtype=K.dtype)).detach()
    model.likelihood.noise = s0.item()
    A = K + s0.item() * torch.eye(n, dtype=K.dtype)
    A_inv = torch.inverse(A)

    # use_prior_a=True: use the prior's actual concentration a (as with map_noise),
    # so the influence equals the exact implicit-function gradient.
    ale_influence = calculate_aleatoric_uncertainty_influence(
        A_inv, y_samples, model, x_samples, use_prior_a=True
    )

    I = torch.eye(n, dtype=K.dtype)

    def sigma_star(h):
        s = s0
        for _ in range(40):
            y_h = y_samples - h * y_samples
            Dh = torch.diag(h)
            K_h = K - Dh @ K - K @ Dh
            A_h = K_h + s * I
            A_inv_h = torch.linalg.solve(A_h, I)
            alpha1 = A_inv_h @ y_h
            alpha2 = A_inv_h @ alpha1
            g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv_h) + (a - 1.0) / s - b
            A_inv2 = A_inv_h @ A_inv_h
            g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv2) - (a - 1.0) / (s * s)
            s = s - g / g_prime
        return s

    h0 = torch.zeros(n, dtype=K.dtype, requires_grad=True)
    grad, = torch.autograd.grad(sigma_star(h0), h0)

    err = (grad - ale_influence).abs().max().item()
    print("aleatoric gradient max abs err:", err)
    assert err < 1e-6, f"aleatoric influence disagrees with autodiff gradient: {err:.3e}"


def test_epistemic_hessian_matches_finite_differences():
    model, x_samples, y_samples, x_test, K, A_inv = _make_problem()
    n = x_samples.shape[0]

    H = calculate_epistemic_hessian(model, x_samples, x_test)

    eps = 1e-4
    H_fd = torch.zeros((n, n), dtype=K.dtype)
    for j in range(n):
        h_plus = torch.zeros(n, dtype=K.dtype)
        h_plus[j] = eps
        h_minus = torch.zeros(n, dtype=K.dtype)
        h_minus[j] = -eps

        with torch.no_grad():
            noise_level = model.likelihood.noise.item()
            K_tx = model.covar_module(x_test, x_samples).evaluate()

        def grad_at(h_val):
            h_req = h_val.clone().requires_grad_(True)
            I = torch.eye(n, dtype=K.dtype)
            grad_sum = torch.zeros(n, dtype=K.dtype)
            for i in range(x_test.shape[0]):
                Dh = torch.diag(h_req)
                K_h = K - Dh @ K - K @ Dh
                A_h = K_h + noise_level * I
                k_h = K_tx[i] - h_req * K_tx[i]
                F = k_h @ torch.linalg.solve(A_h, k_h)
                g, = torch.autograd.grad(F, h_req, create_graph=False)
                grad_sum = grad_sum + (-g)
            return (grad_sum / x_test.shape[0]).detach()

        g_plus = grad_at(h_plus)
        g_minus = grad_at(h_minus)
        H_fd[:, j] = (g_plus - g_minus) / (2 * eps)

    H_fd = 0.5 * (H_fd + H_fd.T)

    abs_err = (H - H_fd).abs().max().item()
    denom = max(H_fd.abs().max().item(), 1.0)
    rel_err = abs_err / denom
    print("epistemic Hessian vs finite-diff: max_abs_err", abs_err, "max_rel_err", rel_err)
    assert rel_err < 1e-4, f"epistemic Hessian mismatch vs finite differences: rel_err={rel_err:.3e}"


def test_aleatoric_hessian_matches_finite_differences():
    model, x_samples, y_samples, x_test, K, A_inv = _make_problem()
    n = x_samples.shape[0]

    a = _get_noise_prior(model.likelihood).concentration.item()
    b = _get_noise_prior(model.likelihood).rate.item()
    noise_level = model.likelihood.noise.item()
    s0 = _solve_map_noise(K, y_samples, a, b, torch.tensor(noise_level, dtype=K.dtype)).detach()
    model.likelihood.noise = s0.item()

    H = calculate_aleatoric_hessian(y_samples, model, x_samples, n_newton_steps=25)

    I = torch.eye(n, dtype=K.dtype)

    def sigma_star(h):
        s = s0
        for _ in range(40):
            y_h = y_samples - h * y_samples
            Dh = torch.diag(h)
            K_h = K - Dh @ K - K @ Dh
            A_h = K_h + s * I
            A_inv_h = torch.linalg.solve(A_h, I)
            alpha1 = A_inv_h @ y_h
            alpha2 = A_inv_h @ alpha1
            g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv_h) + (a - 1.0) / s - b
            A_inv2 = A_inv_h @ A_inv_h
            g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv2) - (a - 1.0) / (s * s)
            s = s - g / g_prime
        return s

    def grad_at(h_val):
        h_req = h_val.clone().requires_grad_(True)
        g, = torch.autograd.grad(sigma_star(h_req), h_req)
        return g.detach()

    eps = 1e-4
    H_fd = torch.zeros((n, n), dtype=K.dtype)
    for j in range(n):
        h_plus = torch.zeros(n, dtype=K.dtype)
        h_plus[j] = eps
        h_minus = torch.zeros(n, dtype=K.dtype)
        h_minus[j] = -eps
        g_plus = grad_at(h_plus)
        g_minus = grad_at(h_minus)
        H_fd[:, j] = (g_plus - g_minus) / (2 * eps)

    H_fd = 0.5 * (H_fd + H_fd.T)

    abs_err = (H - H_fd).abs().max().item()
    denom = max(H_fd.abs().max().item(), 1.0)
    rel_err = abs_err / denom
    print("aleatoric Hessian vs finite-diff: max_abs_err", abs_err, "max_rel_err", rel_err)
    assert rel_err < 1e-3, f"aleatoric Hessian mismatch vs finite differences: rel_err={rel_err:.3e}"


def test_joint_curvature_bounds_hessian_matches_finite_differences():
    """compute_joint_negJ_hessian (used by curvature_bounds.calculate_restricted_curvature_bounds)
    against an independently-built finite-difference Hessian of the same
    -J(h) = mean_x F(x,h,sigma(h)) - sigma(h) objective (extended with y(h))."""
    model, x_samples, y_samples, x_test, K, A_inv = _make_problem(n=6, seed=3)
    n = x_samples.shape[0]

    a = _get_noise_prior(model.likelihood).concentration.item()
    b = _get_noise_prior(model.likelihood).rate.item()
    noise_level = model.likelihood.noise.item()

    with torch.no_grad():
        K_tx = model.covar_module(x_test, x_samples).evaluate()

    I = torch.eye(n, dtype=K.dtype)
    s_init = torch.tensor(noise_level, dtype=K.dtype)

    def neg_J_mean(h, n_steps=25):
        y_h = y_samples - h * y_samples
        Dh = torch.diag(h)
        K_h = K - Dh @ K - K @ Dh
        s = s_init
        for _ in range(n_steps):
            A_h = K_h + s * I
            A_inv_h = torch.linalg.solve(A_h, I)
            alpha1 = A_inv_h @ y_h
            alpha2 = A_inv_h @ alpha1
            g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv_h) + (a - 1.0) / s - b
            A_inv2 = A_inv_h @ A_inv_h
            g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv2) - (a - 1.0) / (s * s)
            s = s - g / g_prime
        A_h = K_h + s * I
        k_h_all = K_tx - h.unsqueeze(0) * K_tx
        sol = torch.linalg.solve(A_h, k_h_all.T)
        F_mean = (k_h_all.T * sol).sum(dim=0).mean()
        return F_mean - s

    def grad_at(h_val):
        h_req = h_val.clone().requires_grad_(True)
        g, = torch.autograd.grad(neg_J_mean(h_req), h_req)
        return g.detach()

    eps = 1e-4
    H_fd = torch.zeros((n, n), dtype=K.dtype)
    for j in range(n):
        h_plus = torch.zeros(n, dtype=K.dtype)
        h_plus[j] = eps
        h_minus = torch.zeros(n, dtype=K.dtype)
        h_minus[j] = -eps
        H_fd[:, j] = (grad_at(h_plus) - grad_at(h_minus)) / (2 * eps)
    H_fd = 0.5 * (H_fd + H_fd.T)

    H = compute_joint_negJ_hessian(model, x_samples, y_samples, x_test)

    abs_err = (H - H_fd).abs().max().item()
    rel_err = abs_err / max(H_fd.abs().max().item(), 1.0)
    print("joint -J Hessian vs finite-diff: max_abs_err", abs_err, "max_rel_err", rel_err)
    assert rel_err < 1e-3, f"joint -J Hessian mismatch vs finite differences: rel_err={rel_err:.3e}"


if __name__ == "__main__":
    test_epistemic_gradient_matches_influence_function()
    test_aleatoric_gradient_matches_influence_function()
    test_epistemic_hessian_matches_finite_differences()
    test_aleatoric_hessian_matches_finite_differences()
    test_joint_curvature_bounds_hessian_matches_finite_differences()
    print("test_curvature_derivatives passed")
