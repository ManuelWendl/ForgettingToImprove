# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch

from . import termination_fns


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.cartpole(act, next_obs)).float().view(-1, 1)


def cartpole_pets(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2
    goal_pos = torch.tensor([0.0, 0.6]).to(next_obs.device)
    x0 = next_obs[:, :1]
    theta = next_obs[:, 1:2]
    ee_pos = torch.cat([x0 - 0.6 * theta.sin(), -0.6 * theta.cos()], dim=1)
    obs_cost = torch.exp(-torch.sum((ee_pos - goal_pos) ** 2, dim=1) / (0.6**2))
    act_cost = -0.01 * torch.sum(act**2, dim=1)
    return (obs_cost + act_cost).view(-1, 1)


def inverted_pendulum(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    return (~termination_fns.inverted_pendulum(act, next_obs)).float().view(-1, 1)


def inverted_pendulum_sparse(
    act: torch.Tensor,
    next_obs: torch.Tensor,
    threshold: float = 0.2,
) -> torch.Tensor:
    """Sparse reward for Pendulum: 1 when |theta| <= *threshold*.

    Parameters
    ----------
    threshold : float
        angle radius (in radians) around vertical that constitutes the
        "goal" region.  Smaller values make the problem harder / sparser.
    """
    assert len(next_obs.shape) == len(act.shape) == 2

    theta = next_obs[:, 1]
    in_goal = (theta.abs() <= threshold)
    return in_goal.float().view(-1, 1)


# ---------------------------------------------------------------------------
# cartpole swingup sparse reward


def cartpole_swingup_sparse(
    act: torch.Tensor,
    next_obs: torch.Tensor,
    threshold: float = 0.2,
    x_threshold: float = 0.5,
) -> torch.Tensor:
    """Sparse reward for the DMC cartpole swingup task.

    The DMC observation vector is ``[x, cos(theta), sin(theta), theta_dot,
    x_dot]``.  Upright corresponds to ``theta =~ \u03c0`` (cosine near +1).

    A reward of ``1`` is returned only when **both** conditions hold:

    1. the pole angle is within ``threshold`` radians of upright
    2. the cart's horizontal position ``x`` is within ``x_threshold`` of
       the origin

    The latter ensures the agent cannot earn points by swinging the pole up
    while driving the cart far away from the centre.  Both thresholds are
    configurable via keyword arguments from the environment wrapper.

    The implementation avoids expensive trigonometry by comparing the
    cosine of the angle directly::

        cos(theta) >= cos(threshold)

    which is true exactly when ``|theta| <= threshold`` (upright region).
    """
    assert len(next_obs.shape) == len(act.shape) == 2

    cos_theta = next_obs[:, 1]
    x_pos = next_obs[:, 0]

    # upright when cos≥cos(threshold); cart near origin if required
    cutoff = torch.cos(torch.tensor(threshold, device=cos_theta.device))
    angle_ok = cos_theta >= cutoff          # ← note the “≥” and no minus
    x_ok = x_pos.abs() <= x_threshold
    in_goal = angle_ok & x_ok
    return in_goal.float().view(-1, 1)


def mountain_car(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    position = next_obs[:, 0]
    velocity = next_obs[:, 1]
    done = (position >= 0.45) & (velocity >= 0.0)
    reward = 100.0 * done.float() - 0.1 * act.pow(2).sum(dim=1)
    return reward.view(-1, 1)


def halfcheetah(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == len(act.shape) == 2

    reward_ctrl = -0.1 * act.square().sum(dim=1)
    reward_run = next_obs[:, 0] - 0.0 * next_obs[:, 2].square()
    return (reward_run + reward_ctrl).view(-1, 1)


def pusher(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    goal_pos = torch.tensor([0.45, -0.05, -0.323]).to(next_obs.device)

    to_w, og_w = 0.5, 1.25
    tip_pos, obj_pos = next_obs[:, 14:17], next_obs[:, 17:20]

    tip_obj_dist = (tip_pos - obj_pos).abs().sum(axis=1)
    obj_goal_dist = (goal_pos - obj_pos).abs().sum(axis=1)
    obs_cost = to_w * tip_obj_dist + og_w * obj_goal_dist

    act_cost = 0.1 * (act**2).sum(axis=1)

    return -(obs_cost + act_cost).view(-1, 1)