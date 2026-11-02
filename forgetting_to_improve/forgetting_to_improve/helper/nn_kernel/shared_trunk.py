"""Shared-trunk, multi-output analogue of ``build_jacobian_kernel``.

A last-layer Laplace model's Jacobian features
phi(x) = [h(x), 1] depend only on the trunk h = body(x), never on the output
head -- so for a linear multi-output head with a separable per-output squared
loss, the GGN is exactly block-diagonal across output dimensions (see
``curvature.py``'s docstring). That means one shared-trunk MLP can serve every
output dimension of a vector-valued regression (e.g. an RL dynamics model's
state-delta dims): only the *mean* (which output column) differs per
dimension, not the kernel/feature map. This is strictly cheaper than fitting
``out_dim`` fully independent MLPs (one trunk fit instead of ``out_dim``).

Two building blocks:
    * ``build_shared_jacobian_kernels`` -- fit the shared trunk once, return
      one (JacobianKernel, MLPMeanIndexed) pair per output dimension.
    * ``finetune_shared_mlp`` (used by ``JacobianGPModel.finetune_shared_trunk``
      in forgetful_rl) -- one joint refit of the trunk, called after the
      removal decisions of all dimensions have been combined, so the features
      of all dimensions stay consistent.
"""
import warnings

import torch
import gpytorch

from .model import MLP, get_activation
from .posterior import posterior_cov_mv
from .pushforward import JacobianKernel

__all__ = [
    "MLPMeanIndexed",
    "train_shared_mlp",
    "build_shared_jacobian_kernels",
    "finetune_shared_mlp",
]


class MLPMeanIndexed(gpytorch.means.Mean):
    """m(x) = f(x, theta*)[output_index] -- like ``MLPMean``, but reads one
    column of a shared multi-output head instead of assuming ``out_dim==1``.

    (``MLPMean`` squeezes the last dimension and is only correct for a scalar head.)
    """

    def __init__(self, model: torch.nn.Module, output_index: int):
        super().__init__()
        self.model = model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.output_index = output_index

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        mdtype = next(self.model.parameters()).dtype
        out = self.model(x.to(mdtype))                # (..., n, out_dim)
        return out[..., self.output_index].to(x.dtype)  # (..., n)


def train_shared_mlp(mlp: torch.nn.Module, X, Y, device="cpu", epochs: int = 500, lr: float = 1e-2):
    """Joint multi-output analogue of ``nn_kernel.train_mlp``: fits the shared
    trunk once against every output column together (theta*), then freezes it.

    A separate function rather than a ``train_mlp`` edit: ``train_mlp``
    unconditionally reshapes targets to ``(-1, 1)``, which would silently
    corrupt a genuine ``(n, out_dim)`` multi-output target.
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    Y = torch.as_tensor(Y, dtype=torch.float32, device=device)
    if Y.dim() == 1:
        Y = Y.unsqueeze(-1)

    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(mlp(X), Y)
        loss.backward()
        opt.step()
    mlp.eval()
    return mlp


def build_shared_jacobian_kernels(config: dict, n_dims: int, out_dim: int,
                                   device="cpu", train_x=None, train_y=None):
    """Shared-trunk analogue of ``build_jacobian_kernel``.

    Fits ONE MLP (an ``out_dim``-wide head on a shared trunk) jointly against
    every output column, then builds one ``(JacobianKernel, MLPMeanIndexed)``
    pair per output dimension -- all referencing that single trunk's frozen
    features, since (see module docstring) the kernel/feature map does not
    depend on the output index, only the mean does.

    Args:
        config: same keys as the single-output jacobian kernel config
            (hidden, depth, activation, last_layer, prior_prec, epochs, lr).
        n_dims: input dimensionality.
        out_dim: number of output dimensions (e.g. obs_size [+1 for reward]).
        train_x: (n, n_dims) training inputs.
        train_y: (n, out_dim) training targets.

    Returns:
        (mlp, kernels, means): the shared trunk, and ``out_dim``-long lists of
        ``JacobianKernel`` / ``MLPMeanIndexed``, each pair usable directly as
        a gpytorch ``ExactGP``'s ``covar_module`` / ``mean_module``.
    """
    hidden = config.get("hidden", 64)
    depth = config.get("depth", 2)
    last_layer = config.get("last_layer", True)
    prior_prec = config.get("prior_prec", 1.0)
    activation = get_activation(config.get("activation", "tanh"))

    mlp = MLP(n_dims, hidden=hidden, depth=depth, out_dim=out_dim, activation=activation).to(device)
    if train_x is not None:
        train_shared_mlp(mlp, train_x, train_y, device=device,
                          epochs=config.get("epochs", 500), lr=config.get("lr", 1e-2))
    else:
        warnings.warn("shared jacobian kernel: no train data provided -> using untrained MLP features")

    # Sigma = I (prior pushforward), identical for every output dimension --
    # same convention as build_jacobian_kernel (see its docstring: the
    # data-dependent posterior is realised by gpytorch's own exact-GP
    # conditioning per dimension, not by accumulating the GGN here).
    cov_mv = posterior_cov_mv(curv_mv=None, prior_prec=prior_prec)
    kernels = [JacobianKernel(mlp, cov_mv, last_layer=last_layer) for _ in range(out_dim)]
    means = [MLPMeanIndexed(mlp, output_index=i) for i in range(out_dim)]
    return mlp, kernels, means


def finetune_shared_mlp(mlp: torch.nn.Module, x_kept: torch.Tensor, y_kept: torch.Tensor,
                         epochs: int = 500, lr: float = 1e-2):
    """The "full fine-tuning" variant: a single joint refit of the *entire*
    shared trunk (all output heads at once), continuing from its current
    weights (call ``reset_parameters(mlp)`` first for a from-scratch refit).

    Called once on the samples kept in all output dimensions (see
    ``forgetful_rl``'s ``filter_samples.py``), not per removal: refitting on
    one dimension's targets would change the features of all others.
    """
    if x_kept.numel() == 0 or x_kept.shape[0] < 2:
        return mlp

    for p in mlp.parameters():
        p.requires_grad_(True)
    train_shared_mlp(mlp, x_kept, y_kept, device=next(mlp.parameters()).device,
                      epochs=epochs, lr=lr)
    for p in mlp.parameters():
        p.requires_grad_(False)
    return mlp
