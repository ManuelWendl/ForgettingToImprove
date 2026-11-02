from typing import Optional, Tuple
import functools
import torch
import gpytorch


def _get_noise_prior(likelihood) -> Optional[gpytorch.priors.Prior]:
    """Locate the GammaPrior on likelihood.noise.

    GaussianLikelihood does not expose `.noise_prior` directly -- the prior
    passed to its constructor is registered on the nested noise_covar module
    (`likelihood.noise_covar.noise_prior`). Checking `.noise_prior` first
    keeps this working for likelihoods that DO expose it directly.
    """
    if hasattr(likelihood, 'noise_prior') and likelihood.noise_prior is not None:
        return likelihood.noise_prior
    noise_covar = getattr(likelihood, 'noise_covar', None)
    if noise_covar is not None and getattr(noise_covar, 'noise_prior', None) is not None:
        return noise_covar.noise_prior
    return None


def _sigma_w2_stationarity_residual(K: torch.Tensor, y: torch.Tensor, s: torch.Tensor, a: float, b: float) -> torch.Tensor:
    """g(s) = 0.5 y^T A(s)^-2 y - 0.5 tr(A(s)^-1) + (a-1)/s - b, A(s) = K + sI."""
    n = y.numel()
    I = torch.eye(n, dtype=K.dtype, device=K.device)
    A_inv = torch.linalg.solve(K + s * I, I)
    alpha1 = A_inv @ y
    return 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv) + (a - 1.0) / s - b


def _robust_sigma_w2_seed(
    K: torch.Tensor,
    y: torch.Tensor,
    a: float,
    b: float,
    s_init: float,
    n_bracket_expand: int = 60,
    n_bisect: int = 60,
) -> torch.Tensor:
    """Root of the MAP-II stationarity residual g(s) = 0 by bracketed bisection
    (no_grad), used to seed the differentiable Newton refinement in
    calculate_aleatoric_hessian / compute_joint_negJ_hessian.

    Plain Newton can diverge when s_init is far from the root for the current
    data (the removal loops do not refit the noise between removals);
    bisection cannot, and the caller then runs a few differentiable Newton
    steps from this point.
    """
    dtype, device = K.dtype, K.device
    with torch.no_grad():
        lo = torch.as_tensor(1e-6, dtype=dtype, device=device)
        hi = torch.as_tensor(max(float(s_init) * 2.0, 1.0), dtype=dtype, device=device)
        g_lo = _sigma_w2_stationarity_residual(K, y, lo, a, b)
        g_hi = _sigma_w2_stationarity_residual(K, y, hi, a, b)

        found_bracket = torch.sign(g_lo) != torch.sign(g_hi)
        expand_iters = 0
        while not found_bracket and expand_iters < n_bracket_expand:
            hi = hi * 2.0
            g_hi = _sigma_w2_stationarity_residual(K, y, hi, a, b)
            found_bracket = torch.sign(g_lo) != torch.sign(g_hi)
            expand_iters += 1

        if not found_bracket:
            # g(s) does not change sign in the searched range (the prior is too
            # weak to guarantee a stationary point): fall back to s_init; the
            # caller's damped Newton refinement stays bounded.
            return torch.as_tensor(float(s_init), dtype=dtype, device=device)

        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            g_mid = _sigma_w2_stationarity_residual(K, y, mid, a, b)
            if torch.sign(g_mid) == torch.sign(g_lo):
                lo, g_lo = mid, g_mid
            else:
                hi, g_hi = mid, g_mid
        return 0.5 * (lo + hi)


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
        K_tx = kernel(x_test, x_samples).evaluate()     # (n_test,n_train)
        K = kernel(x_samples, x_samples).evaluate()     # (n_train,n_train)

    # weights (1 - h), evaluated at h = 0
    w = 1.0 - 0

    # weighted train-test kernel
    K_tx_h = K_tx * w

    # alpha = A^{-1} k_{S,x}
    alpha = A_inv @ K_tx_h.T                            # (n_train,n_test)

    # KD alpha = K D alpha
    KD_alpha = K @ (w * alpha)

    cross = (2.0 * alpha * KD_alpha).mean(dim=1)
    u = 2.0 * (K_tx_h.T * alpha).mean(dim=1) - cross

    factor = (
        1.0
        + (K_tx_h @ A_inv @ A_inv @ K_tx_h.T).diag().mean()
    )
    return u, factor


def _linear_removal_kernel(h: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """K(S,h,S) = K - diag(h)K - K diag(h): the first-order RKHS removal path.

    Exact in the cross term k(x,h,S) = k(x,S) - h*k(x,S); linearized for the
    training-training block K(S,S) (the O(h_i h_j) term diag(h)Kdiag(h) is
    dropped), as in the derivation in the paper appendix. Checked in
    tests/test_curvature_derivatives.py.
    """
    Dh = torch.diag(h)
    return K - Dh @ K - K @ Dh


def calculate_epistemic_hessian(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    x_test: torch.Tensor,
    noise_level: Optional[float] = None,
) -> torch.Tensor:
    """
    Hessian, w.r.t. the RKHS removal directions h, of the epistemic uncertainty

        sigma_epi^2(x; h) = k(x,x) - k(x,h,S)^T A(h)^{-1} k(x,h,S),

    averaged over x_test and evaluated at h=0 via autodiff (matches the sign
    convention of calculate_epistemic_uncertainty_influence, whose returned
    gradient is d(sigma_epi^2)/dh -- i.e. this Hessian's diagonal/quadratic
    form should be read the same way: positive means removing that
    direction *increases* epistemic uncertainty).

    Removal path (same as calculate_epistemic_uncertainty_influence):
        k(x,h,S) = k(x,S) - h ⊙ k(x,S)
        A(h)     = K(h) + sigma_w^2 I,  K(h) = K - diag(h)K - K diag(h)

    sigma_w^2 is held fixed while differentiating w.r.t. h (matching the
    influence function, which also does not re-solve the MAP-II noise
    estimate) -- see calculate_aleatoric_hessian for the sigma_w^2(h)
    contribution.

    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        x_test: Test inputs (n_test, d)
        noise_level: sigma_w^2 to hold fixed (defaults to model.likelihood.noise)

    Returns:
        H: Epistemic Hessian (n_train, n_train), symmetrized
    """
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(-1)
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)

    kernel = model.covar_module
    n_train = x_samples.shape[0]

    if noise_level is None:
        if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
            noise_level = model.likelihood.noise.item()
        else:
            noise_level = 1e-8

    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()
        K_tx = kernel(x_test, x_samples).evaluate()  # (n_test, n_train)

    I = torch.eye(n_train, dtype=K.dtype, device=K.device)

    def rkhs_fit_term(h: torch.Tensor, k0_row: torch.Tensor) -> torch.Tensor:
        # F(h) = k(x,h,S)^T A(h)^{-1} k(x,h,S); sigma_epi^2(x;h) = k(x,x) - F(h)
        A_h = _linear_removal_kernel(h, K) + noise_level * I
        k_h = k0_row - h * k0_row
        sol = torch.linalg.solve(A_h, k_h)
        return k_h @ sol

    H_sum = torch.zeros((n_train, n_train), dtype=K.dtype, device=K.device)
    for i in range(x_test.shape[0]):
        h0 = torch.zeros(n_train, dtype=K.dtype, device=K.device, requires_grad=True)
        H_F = torch.autograd.functional.hessian(
            functools.partial(rkhs_fit_term, k0_row=K_tx[i]), h0
        )
        H_sum = H_sum - H_F  # d^2(sigma_epi^2)/dh^2 = -d^2 F/dh^2

    H = H_sum / x_test.shape[0]
    return 0.5 * (H + H.T)


def calculate_aleatoric_uncertainty_influence(
    A_inv: torch.Tensor, 
    y_samples: torch.Tensor,
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    use_prior_a: bool = False,
) -> torch.Tensor:
    """
    Calculate the influence of each training point on the MLE noise estimate sigma_w^2.
    
    Args:
        A_inv: Inverse of the covariance matrix (K + sigma_w^2 I)^-1, shape (N, N)
        y_samples: Training targets, shape (N,)
        model: The fitted GP model
        x_samples: Training inputs, shape (N, D)
        use_prior_a: Use the noise prior's actual concentration a in the
            -(a-1)/sigma_w^4 term instead of the hard-coded a=1.1 (needed for
            the criterion to equal dJ/dh when a != 1.1)
        
    Returns:
        aleatoric_influence: d(sigma_w^2) / d(h_i) for each i, shape (N,)
    """

    kernel = model.covar_module
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)

    with torch.no_grad():
        # K is the N x N covariance matrix (without noise)
        K = kernel(x_samples, x_samples).evaluate()

    # Pre-calculate common terms
    alpha = A_inv @ y_samples                 # A^-1 y
    A_inv2 = A_inv @ A_inv                    # A^-2
    A_inv3_y = A_inv2 @ alpha                 # A^-3 y
    
    f_hat = K @ alpha

    A_inv2_y = A_inv @ alpha 
    
    K_Ainv2_y = K @ A_inv2_y
    
    Enum_1 = alpha * K_Ainv2_y + f_hat * A_inv2_y
    
    Enum_2 = torch.sum(K * A_inv2.t(), dim=1)

    Enum_target = y_samples * A_inv2_y

    Denom_1 = torch.dot(y_samples, A_inv3_y)
    
    Denom_2 = torch.trace(A_inv2)
    
    H = - Denom_1 + 0.5 * Denom_2

    # Gamma-prior term of dD/d(sigma_w^2): -(a-1)/sigma_w^4 (paper appendix).
    # By default a = 1.1 (the concentration of the default noise prior); with
    # use_prior_a, a is read from the model's noise prior.
    prior_contr = 0.0
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
        a = 1.0
        noise_prior = _get_noise_prior(model.likelihood)
        if noise_prior is not None and hasattr(noise_prior, 'concentration'):
            a = noise_prior.concentration.item()
        prior_contr = ((a if use_prior_a else 1.1) - 1) / noise_level ** 2

    return - (Enum_1 - Enum_2 - Enum_target) / (H - prior_contr)

def calculate_aleatoric_hessian(
    y_samples: torch.Tensor,
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    n_newton_steps: int = 8,
    max_step_fraction: float = 0.5,
) -> torch.Tensor:
    """
    Hessian, w.r.t. the RKHS removal directions h, of the MAP-II noise
    estimate sigma_w^2(h), evaluated at h=0 via autodiff through a
    differentiable Newton refinement of the stationarity condition
    D(sigma_w^2, h) = 0, with D = d/d(sigma_w^2) [log ML + log Gamma-prior]
    = 1/2 y^T A^-2 y - 1/2 tr(A^-1) + (a-1)/sigma_w^2 - b.

    Removal path (matches calculate_aleatoric_uncertainty_influence, which
    is the gradient of this same sigma_w^2(h)):
        K(h) = K - diag(h)K - K diag(h)
        y(h) = y - h ⊙ y
        A(h) = K(h) + sigma_w^2 I

    Numerics: the noise is not refit between removals, so the starting value
    can be far from the root for the current samples. A no_grad bisection
    (_robust_sigma_w2_seed) first gets close to a root; the differentiable
    refinement then uses damped Newton steps (capped at max_step_fraction * s,
    s stays positive).

    Args:
        y_samples: Training targets (n_train,)
        model: The fitted GP model (kernel + likelihood/noise prior)
        x_samples: Training inputs (n_train, d)
        n_newton_steps: Differentiable Newton steps tracking sigma_w^2(h)
        max_step_fraction: Per-step trust-region cap, as a fraction of the
            current s (keeps s within [0.5, 1.5] x its previous value)

    Returns:
        H: Aleatoric Hessian d^2(sigma_w^2)/dh^2 (n_train, n_train), symmetrized
    """
    kernel = model.covar_module
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    n_train = x_samples.shape[0]

    with torch.no_grad():
        K = kernel(x_samples, x_samples).evaluate()

    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
    else:
        noise_level = 1e-8

    a, b = 1.0, 0.0
    noise_prior = _get_noise_prior(model.likelihood) if hasattr(model, 'likelihood') else None
    if noise_prior is not None and hasattr(noise_prior, 'concentration'):
        a = noise_prior.concentration.item()
        b = noise_prior.rate.item()

    I = torch.eye(n_train, dtype=K.dtype, device=K.device)
    s_seed = _robust_sigma_w2_seed(K, y_samples, a, b, noise_level).detach()

    def sigma_star(h: torch.Tensor) -> torch.Tensor:
        s = s_seed
        for _ in range(n_newton_steps):
            y_h = y_samples - h * y_samples
            A_h = _linear_removal_kernel(h, K) + s * I
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
        return s

    h0 = torch.zeros(n_train, dtype=K.dtype, device=K.device, requires_grad=True)
    H = torch.autograd.functional.hessian(sigma_star, h0)
    return 0.5 * (H + H.T)


def solve_map_noise(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    n_newton_steps: int = 8,
    max_step_fraction: float = 0.5,
) -> float:
    """MAP-II noise sigma_w^2 for the current (x_samples, y_samples).

    Same bisection seed + damped Newton refinement as compute_joint_negJ_hessian
    uses at h=0, so setting model.likelihood.noise to this value makes
    calculate_uncertainty_influence's criterion equal to the autodiff dJ/dh
    that the curvature analysis is based on.
    """
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)
    with torch.no_grad():
        K = model.covar_module(x_samples, x_samples).evaluate()
        n = K.shape[0]
        I = torch.eye(n, dtype=K.dtype, device=K.device)

        a, b = 1.0, 0.0
        noise_prior = _get_noise_prior(model.likelihood)
        if noise_prior is not None and hasattr(noise_prior, 'concentration'):
            a = noise_prior.concentration.item()
            b = noise_prior.rate.item()

        s = _robust_sigma_w2_seed(K, y_samples, a, b, model.likelihood.noise.item())
        for _ in range(n_newton_steps):
            A_inv = torch.linalg.solve(K + s * I, I)
            alpha1 = A_inv @ y_samples
            alpha2 = A_inv @ alpha1
            g = 0.5 * torch.dot(alpha1, alpha1) - 0.5 * torch.trace(A_inv) + (a - 1.0) / s - b
            g_prime = -torch.dot(alpha1, alpha2) + 0.5 * torch.trace(A_inv @ A_inv) - (a - 1.0) / (s * s)
            g_prime_safe = torch.sign(g_prime + 1e-30) * torch.clamp(g_prime.abs(), min=1e-6)
            step = torch.clamp(g / g_prime_safe, min=-max_step_fraction * s, max=max_step_fraction * s)
            s = s - step
    return s.item()


def calculate_uncertainty_influence(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    use_prior_a: bool = False,
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
    try:
        A_inv = torch.inverse(A)
    except Exception as e:
        print({e},' Added jitter')
        A_inv = torch.inverse(A + 1e-4 * torch.eye(len(x_samples)))

    epistemic_influence, factor = (
        calculate_epistemic_uncertainty_influence(A_inv, model, x_samples, x_test)
    )
    aleatoric_influence = calculate_aleatoric_uncertainty_influence(
        A_inv, y_samples, model, x_samples, use_prior_a=use_prior_a
    )

    return epistemic_influence, aleatoric_influence, factor