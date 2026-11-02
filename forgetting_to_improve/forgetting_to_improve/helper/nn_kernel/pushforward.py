"""Pushforward stage — mirrors laplax's ``Pushforward.LINEAR``.

The linear pushforward propagates the parameter-space covariance Sigma to
function space, giving the predictive ``N(f(x, theta*), J(x) Sigma J(x')^T)``.

Two objects implement this as a GPyTorch model:
    * ``JacobianKernel``  -> the function-space covariance k(x, x') = J Sigma J^T
    * ``MLPMean``         -> the linearisation centre m(x) = f(x, theta*)

Because linearised Laplace is Bayesian linear regression in the Jacobian
features, feeding the PRIOR kernel (Sigma = I) plus this mean into an exact GP
and conditioning on data reproduces the LINEAR-pushforward posterior via
Woodbury — so no P x P inverse is ever materialised here.

Feature map (Jacobian of the network output w.r.t. params):
    * last_layer=True : phi(x) = [h(x), 1]      (exact, cheap; frozen features)
    * last_layer=False: phi(x) = d f(x)/d theta (full torch.func Jacobian)
"""
import gpytorch
import torch
from torch.func import functional_call, jacrev, vmap


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1) if x.dim() == 1 else x


class JacobianKernel(gpytorch.kernels.Kernel):
    """k(x, x') = phi(x) Sigma phi(x')^T, with Sigma supplied as a cov_mv operator."""

    def __init__(self, model, cov_mv, last_layer: bool = True,
                 params: dict = None, **kwargs):
        super().__init__(**kwargs)
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.cov_mv = cov_mv                     # posterior/prior covariance operator
        self.last_layer = last_layer
        self.params = params or {k: v.detach() for k, v in model.named_parameters()}

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """Jacobian features phi(x), shape (..., n, F). Supports arbitrary leading
        batch dims (BoTorch passes (b, q, d)) and stays differentiable w.r.t. x so
        gradient-based acquisition optimisation works. The MLP params are frozen
        (requires_grad=False), so no parameter grads are accumulated here.
        """
        x = _as_2d(x)
        mdtype = next(self.model.parameters()).dtype
        if self.last_layer:
            h = self.model.features(x.to(mdtype))                # (..., n, H)
            ones = torch.ones(*h.shape[:-1], 1, dtype=h.dtype, device=h.device)
            phi = torch.cat([h, ones], dim=-1)                   # (..., n, H+1)
        else:
            flat = x.reshape(-1, x.shape[-1])

            def f(params, xi):
                return functional_call(self.model, params, (xi.unsqueeze(0).to(mdtype),)).reshape(())

            jac = vmap(jacrev(f), in_dims=(None, 0))(self.params, flat)
            phi = torch.cat([j.reshape(flat.shape[0], -1) for j in jac.values()], dim=-1)
            phi = phi.reshape(*x.shape[:-1], -1)
        return phi.to(x.dtype)

    def forward(self, x1, x2, diag: bool = False, **params):
        phi1 = self._features(x1)
        same = (x2 is x1) or (x1.shape == x2.shape and torch.equal(x1, x2))
        phi2 = phi1 if same else self._features(x2)
        # Apply Sigma as an operator on the feature axis, batch-safely.
        cphi2 = self.cov_mv(phi2.transpose(-1, -2)).transpose(-1, -2)   # (..., n2, F)
        if diag:
            return (phi1 * cphi2).sum(-1)
        return phi1 @ cphi2.transpose(-1, -2)                            # (..., n1, n2)


class MLPMean(gpytorch.means.Mean):
    """m(x) = f(x, theta*) — the linearisation centre of the linear pushforward."""

    def __init__(self, model):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Batch-safe (..., n, d) -> (..., n); differentiable w.r.t. x (params frozen).
        x = _as_2d(x)
        mdtype = next(self.model.parameters()).dtype
        return self.model(x.to(mdtype)).squeeze(-1).to(x.dtype)
