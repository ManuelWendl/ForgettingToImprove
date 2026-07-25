from typing import Optional, Tuple
import torch
import gpytorch
from math import comb
from itertools import combinations


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
    # if x_test.ndim == 1:
    #     x_test = x_test.unsqueeze(-1)
    # if x_samples.ndim == 1:
    #     x_samples = x_samples.unsqueeze(-1)
    
    # kernel = model.covar_module
    
    # with torch.no_grad():
    #     K_tx = kernel(x_test, x_samples).evaluate()
    #     K = kernel(x_samples, x_samples).evaluate()
    
    # epistemic_influence = 2 * torch.mean(K_tx.T * (A_inv @ K_tx.T), dim=1)
    # factor = (1 + (K_tx @ A_inv @ A_inv @ K_tx.T).diag().mean())

    # inner_kernel_der = epistemic_influence.clone()
    # for i in range(x_samples.shape[0]):
    #     K_minus = torch.zeros_like(K)
    #     K_minus[i,:] = - K[i,:]
    #     K_minus[:,i] = - K[:,i]
    #     K_minus[i,i] = - 2 * K[i,i]
    #     inner_kernel_der[i] = (K_tx @ A_inv @ K_minus @ A_inv @ K_tx.T).diag().mean()

    # return epistemic_influence + inner_kernel_der, factor
    
    if x_test.ndim == 1:
        x_test = x_test.unsqueeze(-1)
    if x_samples.ndim == 1:
        x_samples = x_samples.unsqueeze(-1)

    kernel = model.covar_module

    with torch.no_grad():
        K_tx = kernel(x_test, x_samples).evaluate()     # (n_test,n_train)
        K = kernel(x_samples, x_samples).evaluate()     # (n_train,n_train)

    # weights (1-h)
    w = 1.0 - 0

    # weighted train-test kernel
    K_tx_h = K_tx * w

    # alpha = A^{-1} k_{S,x}
    alpha = A_inv @ K_tx_h.T                            # (n_train,n_test)

    # KD alpha = K D alpha
    KD_alpha = K @ (w * alpha)

    cross = (2.0 * alpha * KD_alpha).mean(dim=1)
    u = 2.0 * (K_tx_h.T * alpha).mean(dim=1) - cross   # == inner_kernel_der

    factor = (
        1.0
        + (K_tx_h @ A_inv @ A_inv @ K_tx_h.T).diag().mean()
    )
    return u, factor


def calculate_aleatoric_uncertainty_influence(
    A_inv: torch.Tensor, 
    y_samples: torch.Tensor,
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
) -> torch.Tensor:
    """
    Calculate the influence of each training point on the MLE noise estimate sigma_w^2.
    
    Args:
        A_inv: Inverse of the covariance matrix (K + sigma_w^2 I)^-1, shape (N, N)
        y_samples: Training targets, shape (N,)
        model: The fitted GP model
        x_samples: Training inputs, shape (N, D)
        
    Returns:
        aleatoric_influence: d(sigma_w^2) / d(h_i) for each i, shape (N,)
    """

    # kernel = model.covar_module
    # if x_samples.ndim == 1:
    #     x_samples = x_samples.unsqueeze(-1)

    # with torch.no_grad():
    #     # K is the N x N covariance matrix (without noise)
    #     K = kernel(x_samples, x_samples).evaluate()

    # # Pre-calculate common terms
    # alpha = A_inv @ y_samples                 # A^-1 y
    # A_inv2 = A_inv @ A_inv                    # A^-2
    # A_inv3_y = A_inv2 @ alpha                 # A^-3 y
    
    # f_hat = K @ alpha

    # A_inv2_y = A_inv @ alpha 
    
    # K_Ainv2_y = K @ A_inv2_y
    
    # Enum_1 = alpha * K_Ainv2_y + f_hat * A_inv2_y
    
    # Enum_2 = torch.sum(K * A_inv2.t(), dim=1)

    # Denom_1 = torch.dot(y_samples, A_inv3_y)
    
    # Denom_2 = torch.trace(A_inv2)
    
    # H = - Denom_1 + 0.5 * Denom_2

    # prior_contr = 0.0
    # if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
    #     noise_level = model.likelihood.noise.item()
    #     # Fallback to default Gamma(5, 0.05) if prior not found
    #     prior_contr = (2 - 1) / (noise_level ** 2)

    # return (Enum_1 - Enum_2) / (- torch.abs(H) - prior_contr)

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

    # (a - 1) / sigma_w^4 term from d/d(sigma_w^2) of the Gamma(a, b) log-prior;
    # a is read from the model's actual noise prior (falls back to a=1, i.e. no
    # prior contribution, only if the likelihood has none set).
    prior_contr = 0.0
    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
        a = 1.0
        noise_prior = _get_noise_prior(model.likelihood)
        if noise_prior is not None and hasattr(noise_prior, 'concentration'):
            a = noise_prior.concentration.item()
        prior_contr = (2 - 1) / noise_level ** 2

    return - (Enum_1 - Enum_2 - Enum_target) / (H - prior_contr)

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


def map_ii_stationarity_residual(
    A_inv: torch.Tensor,
    y_samples: torch.Tensor,
    model: gpytorch.models.GP,
) -> float:
    """D(sigma_w^2) = d/d(sigma_w^2) [logML(sigma_w^2) + log Gamma-prior(sigma_w^2)].

    Zero exactly at the sigma_w^2 that Propositions 3-4 (and calculate_aleatoric_
    uncertainty_influence's implicit-function-theorem derivation) are taken at.
    D = 1/2 y^T A^-2 y - 1/2 tr(A^-1) + (a-1)/sigma_w^2 - b.
    """
    A_inv2_y = A_inv @ (A_inv @ y_samples)
    D = 0.5 * torch.dot(y_samples, A_inv2_y) - 0.5 * torch.trace(A_inv)

    if hasattr(model, 'likelihood') and hasattr(model.likelihood, 'noise'):
        noise_level = model.likelihood.noise.item()
        noise_prior = _get_noise_prior(model.likelihood)
        if noise_prior is not None and hasattr(noise_prior, 'concentration'):
            a = noise_prior.concentration.item()
            b = noise_prior.rate.item()
            D = D + (a - 1) / noise_level - b

    return D.item()


def calculate_uncertainty_influence(
    model: gpytorch.models.GP,
    x_samples: torch.Tensor,
    y_samples: torch.Tensor,
    x_test: torch.Tensor,
    check_stationarity: bool = False,
    stationarity_tol: float = 1e-2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Calculate the uncertainty (epistemic and aleatoric) of the GP predictions at points X.

    Args:
        model: GPyTorch GP model
        x_samples: Training inputs (n_train, d)
        y_samples: Training targets (n_train,)
        x_test: Test inputs (n_test, d)
        check_stationarity: If True, assert |D(sigma_w^2)| < stationarity_tol, i.e. that
            model.likelihood.noise is (close to) the MAP-II solution the aleatoric-influence
            derivation assumes. Off by default: the sequential/batch removal loops
            (optimize.py) currently do NOT refit noise between removals, so this would
            fail on every call after the first there -- see calculate_aleatoric_uncertainty_
            influence's docstring / the F2I debugging notes for why that matters.
        stationarity_tol: Absolute tolerance on D for the check above.

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

    if check_stationarity:
        D = map_ii_stationarity_residual(A_inv, y_samples, model)
        assert abs(D) < stationarity_tol, (
            f"MAP-II noise not stationary: D(sigma_w^2)={D:.6g} (tol={stationarity_tol}). "
            "model.likelihood.noise is not the sigma_w^2 that solves D=0 under the Gamma "
            "prior for this subset -- aleatoric_influence (v) is evaluated off the point "
            "Propositions 3-4 assume and its magnitude is not meaningful here."
        )

    epistemic_influence, factor = (
        calculate_epistemic_uncertainty_influence(A_inv, model, x_samples, x_test)
    )
    aleatoric_influence = calculate_aleatoric_uncertainty_influence(A_inv, y_samples, model, x_samples)

    return epistemic_influence, aleatoric_influence, factor