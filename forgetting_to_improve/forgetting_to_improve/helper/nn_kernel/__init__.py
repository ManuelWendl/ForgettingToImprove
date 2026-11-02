"""Jacobian (linearised-Laplace) kernel, structured to mirror the laplax pipeline.

Stage map (laplax  ->  this package):
    model_fn(input, params)              ->  model.py      (make_model_fn / MLP)
    GGN/Hessian vector products          ->  curvature.py  (ggn_mv_factory)
    (Curv + tau I)^{-1}                  ->  posterior.py  (posterior_cov_mv)
    Pushforward.LINEAR: J Sigma J^T      ->  pushforward.py(JacobianKernel, MLPMean)
    calibration(tau, sigma) by marg-lik  ->  calibration.py

``build_jacobian_kernel`` is the high-level entry point (laplax's ``laplace(...)``).
The kernel uses Sigma = I / prior_prec (the prior pushforward, ``curv_mv=None``);
passing a GGN curvature gives the posterior through the same operators.
"""
import warnings

import torch
from torch.nn.utils import parameters_to_vector

from .model import MLP, make_model_fn, get_params, get_activation, reset_parameters, identity_init
from .curvature import ggn_mv_factory, ggn_full
from .posterior import posterior_cov_mv
from .pushforward import JacobianKernel, MLPMean
from .calibration import calibrate_marginal_likelihood
from .prior import Prior, UninformedPrior, IdentityPrior, build_prior
# Shared-trunk, multi-output version (see shared_trunk.py), used by forgetful_rl's
# JacobianGPModel to predict all dynamics output dimensions from one MLP.
from .shared_trunk import (
    MLPMeanIndexed,
    train_shared_mlp,
    build_shared_jacobian_kernels,
    finetune_shared_mlp,
)

__all__ = [
    "MLP", "make_model_fn", "get_params", "get_activation", "reset_parameters", "identity_init",
    "ggn_mv_factory", "ggn_full", "posterior_cov_mv",
    "JacobianKernel", "MLPMean", "calibrate_marginal_likelihood",
    "Prior", "UninformedPrior", "IdentityPrior", "build_prior",
    "train_mlp", "build_jacobian_kernel",
    "MLPMeanIndexed", "train_shared_mlp", "build_shared_jacobian_kernels", "finetune_shared_mlp",
]


def train_mlp(mlp, X, y, device="cpu", epochs=500, lr=1e-2, prior="uninformed", weight_decay=0.0):
    """Train the MLP once (theta*), then freeze it. Features stay fixed thereafter.

    ``weight_decay`` (0.0 by default, i.e. no penalty) adds a
    ``-log p(theta) / N`` penalty from ``prior`` ('uninformed' or 'identity')
    to the MSE loss.
    """
    X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if X.dim() == 1:
        X = X.unsqueeze(-1)
    y = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    prior_obj = None
    if weight_decay > 0:
        prior_obj = build_prior(prior, mlp, weight_decay, X.shape[0], device)

    mlp.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(mlp(X), y)
        if prior_obj is not None:
            params = parameters_to_vector(mlp.parameters())
            loss = loss - prior_obj.log_prob(params) / X.shape[0]
        loss.backward()
        opt.step()
    mlp.eval()
    return mlp


def build_jacobian_kernel(config, n_dims, device="cpu", train_x=None, train_y=None):
    """High-level builder (analogue of laplax ``laplace(...)``).

    Returns ``(kernel, mlp)``: the ``JacobianKernel`` (linear pushforward) and the
    frozen MLP (so the caller can build the matching ``MLPMean``).
    """
    hidden = config.get("hidden", 64)
    depth = config.get("depth", 2)
    last_layer = config.get("last_layer", True)
    prior_prec = config.get("prior_prec", 1.0)
    activation = get_activation(config.get("activation", "tanh"))
    prior = config.get("prior", "uninformed")
    weight_decay = config.get("weight_decay", 0.0)

    mlp = MLP(n_dims, hidden=hidden, depth=depth, activation=activation).to(device)
    if prior == "identity":
        # Start MAP training at the identity prior's mode
        identity_init(mlp)
    if train_x is not None:
        train_mlp(mlp, train_x, train_y, device=device,
                  epochs=config.get("epochs", 500), lr=config.get("lr", 1e-2),
                  prior=prior, weight_decay=weight_decay)
    else:
        warnings.warn("jacobian kernel: no train data provided -> using untrained MLP features")

    # Sigma = I  (prior pushforward): curv_mv=None, tau = prior_prec.
    cov_mv = posterior_cov_mv(curv_mv=None, prior_prec=prior_prec)
    kernel = JacobianKernel(mlp, cov_mv, last_layer=last_layer)
    return kernel, mlp
