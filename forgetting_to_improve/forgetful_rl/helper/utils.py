from typing import Union, Any

import numpy as np
import gymnasium as gym
import omegaconf

from ..model.ensemble_model import EnsembleModel


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)
        

def to_np(t):
    if t is None:
        return None
    elif t.nelement() == 0:
        return np.array([])
    else:
        return t.cpu().detach().numpy()
    

def truncated_linear(
    min_x: float, max_x: float, min_y: float, max_y: float, x: float
) -> float:
    """Truncated linear function.

    Implements the following function:
        f1(x) = min_y + (x - min_x) / (max_x - min_x) * (max_y - min_y)
        f(x) = min(max_y, max(min_y, f1(x)))

    If max_x - min_x < 1e-10, then it behaves as the constant f(x) = max_y
    """
    if max_x - min_x < 1e-10:
        return max_y
    if x <= min_x:
        y: float = min_y
    else:
        dx = (x - min_x) / (max_x - min_x)
        dx = min(dx, 1.0)
        y = dx * (max_y - min_y) + min_y
    return y


def complete_agent_cfg(
    env: Union[gym.Env, EnsembleModel], agent_cfg: omegaconf.DictConfig
):
    """Completes an agent's configuration given information from the environment.

    The goal of this function is to completed information about state and action shapes and ranges,
    without requiring the user to manually enter this into the Omegaconf configuration object.

    It will check for and complete any of the following keys:

        - "obs_dim": set to env.observation_space.shape
        - "action_dim": set to env.action_space.shape
        - "action_range": set to max(env.action_space.high) - min(env.action_space.low)
        - "action_lb": set to env.action_space.low
        - "action_ub": set to env.action_space.high

    Note:
        If the user provides any of these values in the Omegaconf configuration object, these
        *will not* be overridden by this function.

    """
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    def _check_and_replace(key: str, value: Any, cfg: omegaconf.DictConfig):
        if key in cfg.keys() and key not in cfg:
            setattr(cfg, key, value)

    _check_and_replace("num_inputs", obs_shape[0], agent_cfg)

    if "obs_dim" in agent_cfg.keys() and "obs_dim" not in agent_cfg:
        agent_cfg.obs_dim = obs_shape[0]
    if "action_dim" in agent_cfg.keys() and "action_dim" not in agent_cfg:
        agent_cfg.action_dim = act_shape[0]
    if "action_range" in agent_cfg.keys() and "action_range" not in agent_cfg:
        agent_cfg.action_range = [
            float(env.action_space.low.min()),
            float(env.action_space.high.max()),
        ]
    if "action_lb" in agent_cfg.keys() and "action_lb" not in agent_cfg:
        agent_cfg.action_lb = env.action_space.low.tolist()
    if "action_ub" in agent_cfg.keys() and "action_ub" not in agent_cfg:
        agent_cfg.action_ub = env.action_space.high.tolist()

    return agent_cfg
    