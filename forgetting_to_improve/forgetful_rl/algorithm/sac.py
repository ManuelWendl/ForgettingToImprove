import os
import wandb

from typing import Optional, cast

import gymnasium as gym
import numpy as np
import omegaconf
import torch
import hydra.utils

from ..helper import types
from ..helper import logging
from ..helper import common
from ..helper import utils
from ..helper import pytorch_sac_pranz24

MBPO_LOG_FORMAT = [
    ("env_step", "S", "int"),
    ("episode_reward", "R", "float"),
] + [
    ("epoch", "E", "int"),
    ("rollout_length", "RL", "int"),
]

class RandomAgent:
    def __init__(self, env: gym.Env):
        self.env = env

    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
        return self.env.action_space.sample()


class SACAgent:
    def __init__(self, sac_learner):
        self.sac_agent = sac_learner

    def act(self, obs: np.ndarray, sample: bool = False, batched: bool = False, **kwargs) -> np.ndarray:
        with torch.no_grad():
            return self.sac_agent.select_action(
                obs, batched=batched, evaluate=not sample
            )


def evaluate(
    env: gym.Env,
    agent: SACAgent,
    num_episodes: int,
) -> float:
    avg_episode_reward = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0
        while not terminated and not truncated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
        avg_episode_reward += episode_reward
    return avg_episode_reward / num_episodes

def train(
    env: gym.Env,
    test_env: gym.Env,
    termination_fn: types.TermFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    utils.complete_agent_cfg(env, cfg.algorithm.agent)
    agent = SACAgent(
        cast(pytorch_sac_pranz24.SAC, hydra.utils.instantiate(
            cfg.algorithm.agent, action_space=env.action_space
        ))
    )

    work_dir = work_dir or os.getcwd()
    # enable_back_compatible to use pytorch_sac agent
    logger = logging.Logger(cfg, work_dir, enable_back_compatible=True)
    logger.register_group(
        "results",
        MBPO_LOG_FORMAT,
        color="green",
        dump_frequency=1,
    )

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    # -------------- Create initial overrides. dataset --------------
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    random_explore = cfg.algorithm.random_initial_explore
    common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        RandomAgent(env) if random_explore else agent,
        {} if random_explore else {"sample": True, "batched": False},
        replay_buffer=replay_buffer,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    updates_made = 0
    env_steps = 0
    best_eval_reward = -np.inf
    epoch = 0
    iteration = 0
    while env_steps < cfg.overrides.num_steps:
        rollout_length = int(
            utils.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
        obs = None
        terminated = False
        truncated = False
        for steps_epoch in range(cfg.overrides.epoch_length):
            if steps_epoch == 0 or terminated or truncated:
                steps_epoch = 0
                obs, _ = env.reset()
                terminated = False
                truncated = False
            # --- Doing env step and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = common.step_env_and_add_to_buffer(
                env, obs, agent, {"sample": True}, replay_buffer
            )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(replay_buffer) < cfg.overrides.sac_batch_size:
                    break  # only update every once in a while

                agent.sac_agent.update_parameters(
                    replay_buffer,
                    cfg.overrides.sac_batch_size,
                    updates_made,
                    logger,
                    reverse_mask=True,
                )
                updates_made += 1

            env_steps += 1
            obs = next_obs

        # ------ Iteration ended (evaluate and save model) ------
        if iteration % cfg.evaluation_interval == 0:
            avg_reward = evaluate(test_env, agent, cfg.algorithm.num_eval_episodes)
            logger.log_data(
                "results",
                {
                    "iteration": iteration,
                    "env_step": env_steps,
                    "eval/episode_reward": avg_reward,
                    "train/rollout_length": rollout_length,
                },
            )
            print(f"Iteration {iteration} eval episode reward: {avg_reward}")
        iteration += 1
    
    # Evaluate at the end of training
    avg_reward = evaluate(test_env, agent, cfg.algorithm.num_eval_episodes)
    logger.log_data(
        "results",
        {
            "iteration": iteration,
            "env_step": env_steps,
            "eval/episode_reward": avg_reward,
            "train/rollout_length": rollout_length,
        },
    )
    print(f"Iteration {iteration} eval episode reward: {avg_reward}")
    wandb.finish()

    return np.float32(best_eval_reward)