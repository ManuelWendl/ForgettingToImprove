"""Weight-space priors for the Jacobian/linearised-Laplace MLP.

Two priors: the zero-mean uninformed prior (its log-prob term is plain L2
weight decay) and the identity prior, which centres square hidden weight
matrices at I so training regularises towards ``||W - I||^2`` instead of
``||W||^2``. Both enter as a ``-log p(theta) / N`` penalty added to the MLP's
training loss in ``train_mlp`` (see ``__init__.py``).
"""
import torch
import torch.distributions as td
import torch.nn as nn


class Prior:
    """Thin wrapper around a torch distribution over the flattened parameters."""

    def __init__(self, distribution):
        self.distribution = distribution

    def log_prob(self, params: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(params)


class UninformedPrior(Prior):
    """Zero-mean Gaussian prior; equivalent to standard L2 weight decay."""

    def __init__(self, num_parameters: int, weight_decay: float,
                 num_training_samples: int, device):
        wd = weight_decay * num_training_samples
        distribution = td.lowrank_multivariate_normal.LowRankMultivariateNormal(
            torch.zeros(num_parameters, device=device),
            torch.zeros(num_parameters, device=device)[:, None],
            torch.ones(num_parameters, device=device) / wd,
            validate_args=True
        )
        super().__init__(distribution)


class IdentityPrior(Prior):
    """Gaussian prior centred at the identity matrix for square hidden weight layers.

    Non-square weights (input/output projections) and all biases use a zero
    mean, matching ``UninformedPrior``. Square hidden weight matrices use
    ``vec(I)`` as mean, so the regulariser penalises ``||W - I||^2`` rather
    than ``||W||^2``.
    """

    def __init__(self, model: nn.Module, num_parameters: int, weight_decay: float,
                 num_training_samples: int, device):
        wd = weight_decay * num_training_samples

        mean = torch.zeros(num_parameters)
        offset = 0
        for param in model.parameters():
            numel = param.numel()
            if param.dim() == 2 and param.shape[0] == param.shape[1]:
                mean[offset:offset + numel] = torch.eye(param.shape[0]).flatten()
            offset += numel

        mean = mean.to(device)
        distribution = td.lowrank_multivariate_normal.LowRankMultivariateNormal(
            mean,
            torch.zeros(num_parameters, device=device)[:, None],
            torch.ones(num_parameters, device=device) / wd,
            validate_args=True
        )
        super().__init__(distribution)


def build_prior(prior_type: str, model: nn.Module, weight_decay: float,
                num_training_samples: int, device) -> Prior:
    """Factory mirroring (a subset of) ``PriorFactory.create_prior``."""
    num_parameters = sum(p.numel() for p in model.parameters())
    if prior_type == "identity":
        return IdentityPrior(model, num_parameters, weight_decay, num_training_samples, device)
    if prior_type == "uninformed":
        return UninformedPrior(num_parameters, weight_decay, num_training_samples, device)
    raise ValueError(f"Unknown prior '{prior_type}'. Options: 'uninformed', 'identity'")
