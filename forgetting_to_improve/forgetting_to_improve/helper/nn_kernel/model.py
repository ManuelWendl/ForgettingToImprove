"""Model stage — mirrors laplax's ``model_fn:(input, params) -> output``.

laplax treats the network as a pure function of an (input, params) pair, with
``params`` an arbitrary PyTree. Here the PyTree is a ``dict`` of named tensors and
``torch.func.functional_call`` plays the role of the framework-agnostic apply.
"""
import torch
import torch.nn as nn
from torch.func import functional_call


class MLP(nn.Module):
    """Plain MLP. ``features(x)`` exposes the penultimate activations h(x) so the
    last-layer Jacobian is simply ``[h(x), 1]`` (weight + bias derivative)."""

    def __init__(self, in_dim: int, hidden: int = 64, depth: int = 2,
                 out_dim: int = 1, activation=nn.Tanh):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), activation()]
            d = hidden
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(d, out_dim)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))


_ACTIVATIONS = {
    "tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU,
    "swish": nn.SiLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU,
    "sigmoid": nn.Sigmoid, "softplus": nn.Softplus,
}


def get_activation(name):
    """Resolve an activation from a config string (case-insensitive) to an nn class.

    Passing None -> Tanh (the default); passing a class/callable returns it as-is.
    The activation shapes the last-layer feature map phi(x)=[h(x),1] and hence the
    kernel: Tanh gives smooth bounded features, ReLU piecewise-linear/rougher ones.
    """
    if name is None:
        return nn.Tanh
    if not isinstance(name, str):
        return name
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. Options: {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[key]


def reset_parameters(model: nn.Module) -> nn.Module:
    """Re-initialise every layer in place, discarding theta*.

    In place because JacobianKernel and MLPMean hold a reference to this same
    module -- rebinding a fresh MLP would leave them on the old weights.
    """
    for m in model.modules():
        if hasattr(m, "reset_parameters"):
            m.reset_parameters()
    return model


def identity_init(model: nn.Module) -> nn.Module:
    """Xavier init for every Linear layer; square weight matrices start at I.

    Matches the IdentityPrior mean (vec(I) for square hidden weights, zero
    elsewhere; see ``prior.py``), so MAP training of the MLP starts exactly at
    the prior mode instead of at a Xavier-random point.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            w = m.weight
            if w.dim() == 2 and w.shape[0] == w.shape[1]:
                with torch.no_grad():
                    w.copy_(torch.eye(w.shape[0]))
            else:
                nn.init.xavier_uniform_(w)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    return model


def make_model_fn(model: nn.Module):
    """Return laplax-style ``model_fn(params, x)`` with ``params`` a dict PyTree."""
    def model_fn(params: dict, x: torch.Tensor) -> torch.Tensor:
        return functional_call(model, params, (x,))
    return model_fn


def get_params(model: nn.Module, trainable_only: bool = False) -> dict:
    """Detach the model parameters into a dict PyTree (θ*)."""
    return {k: v.detach() for k, v in model.named_parameters()
            if (v.requires_grad or not trainable_only)}
