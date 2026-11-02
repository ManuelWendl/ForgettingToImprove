import numpy as np
import omegaconf
import torch
from typing import Tuple, Optional, Union, Dict

# The wrappers below come from the legacy `gym` package (as in the paper's
# experiments); the cartpole environment itself is a gymnasium env.
import gym

from ..env import cartpole_continuous
from ..env import termination_fns as term_functions
from ..env import reward_fns as reward_functions


class RewardTermWrapper(gym.Wrapper):
    """Gym wrapper that overrides reward and/or termination using provided functions.

    The wrapped environment's ``step`` method is called normally, then the
    supplied ``reward_fn`` and ``term_fn`` (if not ``None``) are evaluated with
    the most recent action and next observation.  These functions are expected
    to have signatures similar to those in ``env/reward_fns.py`` /
    ``env/termination_fns.py`` (typically ``(act: Tensor, next_obs: Tensor)``).
    Keyword arguments passed via ``reward_kwargs``/``term_kwargs`` will also
    be forwarded, which allows configuration of thresholds or other parameters.
    """

    def __init__(self, env: gym.Env, reward_fn=None, term_fn=None, *,
                 reward_kwargs=None, term_kwargs=None):
        super().__init__(env)
        self.reward_fn = reward_fn
        self.term_fn = term_fn
        self.reward_kwargs = {} if reward_kwargs is None else reward_kwargs
        self.term_kwargs = {} if term_kwargs is None else term_kwargs

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # convert to tensors with batch dimension
        if self.reward_fn is not None:
            act_t = torch.from_numpy(np.atleast_2d(action)).float()
            obs_t = torch.from_numpy(np.atleast_2d(obs)).float()
            r_t = self.reward_fn(act_t, obs_t, **self.reward_kwargs)
            # r_t may be a tensor of shape (1,1)
            reward = float(r_t.cpu().numpy().squeeze())
        if self.term_fn is not None:
            act_t = torch.from_numpy(np.atleast_2d(action)).float()
            obs_t = torch.from_numpy(np.atleast_2d(obs)).float()
            done_t = self.term_fn(act_t, obs_t, **self.term_kwargs)
            terminated = bool(
                torch.as_tensor(done_t).cpu().numpy().squeeze()
            )
        return obs, reward, terminated, truncated, info



class EnvHandler:
    @staticmethod
    def make_env(
        cfg: Union[Dict, omegaconf.DictConfig],
    ) -> Tuple[gym.Env, callable, Optional[callable]]:
        """
        Creates the environment and maps the correct termination/reward functions.
        """
        if isinstance(cfg, dict):
            cfg = omegaconf.OmegaConf.create(cfg)
        
        env_name = cfg.overrides.env
        
        # --- Environment and reward/termination functions ---
        if env_name == "cartpole_continuous":
            env = cartpole_continuous.CartPoleEnv()
            # No early termination (see cartpole_continuous.py's step()): episodes run
            # for trial_length steps; the reward is 0 outside the balance bounds. Model
            # rollouts use the same no-termination function.
            term_fn = term_functions.no_termination
            reward_fn = reward_functions.cartpole
        elif "dmc___" in env_name:
            # DeepMind Control Suite environment, format dmc___<domain>___<task>
            # (e.g. dmc___cartpole___swingup); requires dm_control.
            from ..helper import dmc2gym
            parts = env_name.split("___")
            domain_name = parts[1]
            task_name = parts[2]
            # action repeat / frame skip via `action_repeat` (or `frame_skip`)
            frame_skip = cfg.overrides.get("action_repeat", cfg.overrides.get("frame_skip", 1))
            env = dmc2gym.make(
                domain_name=domain_name,
                task_name=task_name,
                seed=cfg.seed,
                frame_skip=frame_skip,
                episode_length=cfg.overrides.get("trial_length", 1000)
            )
            term_fn = term_functions.no_termination
            # Optional custom reward (e.g. cartpole_swingup_sparse); default: DMC reward
            reward_fn = None
            if cfg.overrides.get("reward_fn") is not None:
                reward_fn = getattr(
                    reward_functions,
                    cfg.overrides.reward_fn,
                    None,
                )
        else:
            raise ValueError(f"Invalid environment string: {env_name}")

        # --- Standard Wrappers ---
        # DMC environments already have a time limit
        if "dmc___" not in env_name:
            trial_length = cfg.overrides.get("trial_length", 1000)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=trial_length)

        # With learned rewards, no analytic reward function is used
        if cfg.overrides.get("learned_rewards", True):
            reward_fn = None
        # --- Wrap environment if a custom reward or termination is requested ---
        # this ensures both training and evaluation use the override functions
        if reward_fn is not None or term_fn is not None:
            # gather any kwargs for the reward/termination functions
            rw_kwargs = {}
            tw_kwargs = {}
            # thresholds of the sparse swingup reward
            if reward_fn is not None and cfg.overrides.get("sparse_threshold") is not None:
                rw_kwargs["threshold"] = cfg.overrides.sparse_threshold
            if reward_fn is not None and cfg.overrides.get("sparse_x_threshold") is not None:
                rw_kwargs["x_threshold"] = cfg.overrides.sparse_x_threshold
            env = RewardTermWrapper(
                env,
                reward_fn=reward_fn,
                term_fn=term_fn,
                reward_kwargs=rw_kwargs,
                term_kwargs=tw_kwargs,
            )
        # --- Seeding ---
        if cfg.seed is not None:
            env.reset(seed=cfg.seed)
            env.action_space.seed(cfg.seed + 1)
            env.observation_space.seed(cfg.seed + 2)

        return env, term_fn, reward_fn