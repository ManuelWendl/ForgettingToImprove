"""Posterior stage — mirrors laplax's posterior precision operator.

laplax maps a prior precision tau and curvature C to a covariance operator

    (tau, C) |-> [ v |-> (Curv(C) + tau * I)^{-1} v ].

We return the covariance as a callable ``cov_mv(V) -> Sigma V`` so the
pushforward code is unchanged across variants:

    * curv_mv is None  ->  Sigma = (1/tau) I           (PRIOR pushforward; the
                                                         Sigma = I starting point)
    * curv_mv given     ->  Sigma = (Curv + tau I)^{-1}  (FULL variant, GGN Laplace)

The Sigma = I case is therefore not a separate code path — it is exactly the
above formula with ``Curv = 0`` and ``tau = 1``.
"""
import torch


def posterior_cov_mv(curv_mv=None, prior_prec: float = 1.0,
                     n_features: int = None, variant: str = "full"):
    """Build the posterior covariance operator Sigma (as a callable).

    Args:
        curv_mv: matrix-free curvature product (or None for the prior pushforward).
        prior_prec: tau, the isotropic Gaussian prior precision.
        n_features: F, required only when materialising a FULL posterior.
        variant: "full" (materialise + invert) — placeholder for diag/lanczos.

    Returns:
        cov_mv: callable V -> Sigma V, where V is (F,) or (F, m).
    """
    if curv_mv is None:
        # Sigma = (1/tau) I  — prior pushforward (the Sigma = I default with tau=1)
        return lambda V: V / prior_prec

    if variant != "full":
        raise NotImplementedError(f"posterior variant '{variant}' not yet implemented")

    if n_features is None:
        raise ValueError("n_features is required to materialise a FULL posterior")

    eye = torch.eye(n_features)
    # Materialise precision column by column from the mv, then invert once.
    prec = curv_mv(eye) + prior_prec * eye          # (Curv + tau I)
    cov = torch.linalg.inv(prec)                    # Sigma

    def cov_mv(V: torch.Tensor) -> torch.Tensor:
        return cov.to(V.dtype) @ V
    return cov_mv
