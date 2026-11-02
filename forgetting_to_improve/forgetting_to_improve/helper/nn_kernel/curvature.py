"""Curvature stage — mirrors laplax's GGN/Hessian *vector products*.

laplax exposes curvature as a matrix-free ``mv`` callable, computed over an
iterable of data points. For a regression likelihood the GGN is

    GGN = (1/sigma^2) * sum_n J(x_n)^T J(x_n) = (1/sigma^2) * Phi^T Phi,

where ``Phi`` are the (last-layer or full) Jacobian features. We return the GGN
as a callable ``mv(v) -> (Curv) v`` so that downstream code (posterior.py) is
identical whether the curvature is FULL, DIAGONAL, or absent (prior-only).

For the Sigma = I starting point the curvature is simply not accumulated
(``curv_mv=None`` in posterior.py), i.e. the *prior* pushforward.
"""
import torch


def ggn_mv_factory(features: torch.Tensor, sigma_noise: float = 1.0):
    """Matrix-free GGN-vector product from precomputed Jacobian features.

    Args:
        features: Phi of shape (n, F) — J(x_n) stacked over the data.
        sigma_noise: observation noise std (regression likelihood precision 1/sigma^2).

    Returns:
        mv: callable v -> (1/sigma^2) Phi^T (Phi v), matching laplax's GGN-vp.
    """
    inv_var = 1.0 / (sigma_noise ** 2)

    def mv(v: torch.Tensor) -> torch.Tensor:  # v: (F,) or (F, m)
        return inv_var * (features.t() @ (features @ v))
    return mv


def ggn_full(features: torch.Tensor, sigma_noise: float = 1.0) -> torch.Tensor:
    """Dense GGN (laplax CurvApprox.FULL): (1/sigma^2) Phi^T Phi, shape (F, F)."""
    inv_var = 1.0 / (sigma_noise ** 2)
    return inv_var * (features.t() @ features)
