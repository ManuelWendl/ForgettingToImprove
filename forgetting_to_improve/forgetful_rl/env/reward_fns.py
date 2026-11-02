# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_swingup_sparse(
    act: torch.Tensor,
    next_obs: torch.Tensor,
    threshold: float = 0.2,
    x_threshold: float = 0.5,
) -> torch.Tensor:
    """Sparse reward for the DMC cartpole swingup task.

    The DMC observation vector is ``[x, cos(theta), sin(theta), theta_dot,
    x_dot]``, with the pole upright at cos(theta) = +1.

    A reward of ``1`` is returned only when **both** conditions hold:

    1. the pole angle is within ``threshold`` radians of upright
    2. the cart's horizontal position ``x`` is within ``x_threshold`` of
       the origin

    The latter ensures the agent cannot earn points by swinging the pole up
    while driving the cart far away from the centre.  Both thresholds are
    configurable via keyword arguments from the environment wrapper.

    The angle condition is checked as cos(theta) >= cos(threshold).
    """
    assert len(next_obs.shape) == len(act.shape) == 2

    cos_theta = next_obs[:, 1]
    x_pos = next_obs[:, 0]

    cutoff = torch.cos(torch.tensor(threshold, device=cos_theta.device))
    angle_ok = cos_theta >= cutoff
    x_ok = x_pos.abs() <= x_threshold
    in_goal = angle_ok & x_ok
    return in_goal.float().view(-1, 1)
