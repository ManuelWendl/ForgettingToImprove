# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Sequence, Union

import numpy as np
import torch
from torch import nn as nn

from ..helper import types


def truncated_normal_init(m: nn.Module):
    """Initializes the weights of the given module using a truncated normal distribution."""

    if isinstance(m, nn.Linear):
        input_dim = m.weight.data.shape[0]
        stddev = 1 / (2 * np.sqrt(input_dim))
        truncated_normal_(m.weight.data, std=stddev)
        m.bias.data.fill_(0.0)
    if isinstance(m, EnsembleLinearLayer):
        num_members, input_dim, _ = m.weight.data.shape
        stddev = 1 / (2 * np.sqrt(input_dim))
        for i in range(num_members):
            truncated_normal_(m.weight.data[i], std=stddev)
        m.bias.data.fill_(0.0)


# inplace truncated normal function for pytorch.
# credit to https://github.com/Xingyu-Lin/mbpo_pytorch/blob/main/model.py#L64
def truncated_normal_(
    tensor: torch.Tensor, mean: float = 0, std: float = 1
) -> torch.Tensor:
    """Samples from a truncated normal distribution in-place.

    Args:
        tensor (tensor): the tensor in which sampled values will be stored.
        mean (float): the desired mean (default = 0).
        std (float): the desired standard deviation (default = 1).

    Returns:
        (tensor): the tensor with the stored values. Note that this modifies the input tensor
            in place, so this is just a pointer to the same object.
    """
    torch.nn.init.normal_(tensor, mean=mean, std=std)
    while True:
        cond = torch.logical_or(tensor < mean - 2 * std, tensor > mean + 2 * std)
        bound_violations = torch.sum(cond).item()
        if bound_violations == 0:
            break
        tensor[cond] = torch.normal(
            mean, std, size=(bound_violations,), device=tensor.device
        )
    return tensor


class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.rand(self.num_members, self.in_size, self.out_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.rand(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight)
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite


class Normalizer:
    """Class that keeps a running mean and variance and normalizes data accordingly.

    The statistics kept are stored in torch tensors.

    Args:
        in_size (int): the size of the data that will be normalized.
        device (torch.device): the device in which the data will reside.
        dtype (torch.dtype): the data type to use for the normalizer.
    """

    _STATS_FNAME = "env_stats.pickle"

    def __init__(self, in_size: int, device: torch.device, dtype=torch.float32):
        self.mean = torch.zeros((1, in_size), device=device, dtype=dtype)
        self.std = torch.ones((1, in_size), device=device, dtype=dtype)
        self.eps = 1e-12 if dtype == torch.double else 1e-5
        self.device = device

    def update_stats(self, data: types.TensorType):
        """Updates the stored statistics using the given data.

        Equivalent to `self.stats.mean = data.mean(0) and self.stats.std = data.std(0)`.

        Args:
            data (np.ndarray or torch.Tensor): The data used to compute the statistics.
        """
        assert data.ndim == 2 and data.shape[1] == self.mean.shape[1]
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).to(self.device)
        self.mean = data.mean(0, keepdim=True)
        self.std = data.std(0, keepdim=True)
        self.std[self.std < self.eps] = 1.0

    def normalize(self, val: Union[float, types.TensorType]) -> torch.Tensor:
        """Normalizes the value according to the stored statistics.

        Equivalent to (val - mu) / sigma, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to normalize.

        Returns:
            (torch.Tensor): The normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return (val - self.mean) / self.std

    def denormalize(self, val: Union[float, types.TensorType]) -> torch.Tensor:
        """De-normalizes the value according to the stored statistics.

        Equivalent to sigma * val + mu, where mu and sigma are the stored mean and
        standard deviation, respectively.

        Args:
            val (float, np.ndarray or torch.Tensor): The value to de-normalize.

        Returns:
            (torch.Tensor): The de-normalized value.
        """
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).to(self.device)
        return self.std * val + self.mean


def to_tensor(x: types.TensorType):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    raise ValueError("Input must be torch.Tensor or np.ndarray.")