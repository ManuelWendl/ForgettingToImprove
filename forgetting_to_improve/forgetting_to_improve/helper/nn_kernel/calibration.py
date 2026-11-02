"""Calibration stage — mirrors laplax's ``calibration(...)``.

laplax tunes the prior precision tau and observation noise sigma by maximising
the marginal likelihood (or a downstream metric). Under the GP-dual view here
that is exactly fitting the ScaleKernel outputscale (= 1/tau) and the Gaussian
likelihood noise (= sigma^2) with the Jacobian features held frozen.

Optional utility; the experiments in this repository do not use it.
"""
import gpytorch
import torch


def calibrate_marginal_likelihood(model, likelihood, train_x, train_y,
                                  iters: int = 100, lr: float = 0.1):
    """Fit outputscale (1/tau) and noise (sigma^2) by marginal likelihood.

    Only the ScaleKernel outputscale and likelihood noise are optimised; the
    frozen MLP features (and hence the kernel geometry) are untouched.
    """
    model.train()
    likelihood.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for _ in range(iters):
        opt.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        opt.step()
    model.eval()
    likelihood.eval()
    return model, likelihood
