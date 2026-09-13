import os
import wandb

from typing import Optional, cast

import gymnasium as gym
import numpy as np
import omegaconf
import torch
import hydra.utils

from .sac import SACAgent
from ..data.buffer import ReplayBuffer
from ..helper import types
from ..helper import logging
from ..helper import common
from ..helper import utils
from ..helper import pytorch_sac_pranz24
import torch.nn as nn
from ..forgetting.target_region import TargetRegion
from ..forgetting.filter_samples import filter_replay_buffer

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


def rollout_model_and_populate_sac_buffer(
    dynamics_model: nn.Module,
    replay_buffer: ReplayBuffer,
    agent: SACAgent,
    sac_buffer: ReplayBuffer,
    sac_samples_action: bool,
    rollout_horizon: int,
    total_batch_size: int,
    batch_size: int,
    model_updates: int,
    reward_fn: Optional[types.TermFnType] = None,
):
    if total_batch_size % batch_size != 0:
        raise RuntimeError("total_batch_size must be divisible by batch_size.")
    for _ in range(total_batch_size // batch_size):
        batch = replay_buffer.sample(batch_size)
        initial_obs, *_ = cast(types.TransitionBatch, batch).astuple()
        model_state = dynamics_model.reset(
            initial_obs_batch=cast(np.ndarray, initial_obs),
            return_as_np=True,
        )
        accum_dones = np.zeros(initial_obs.shape[0], dtype=bool)
        obs = initial_obs
        for _ in range(rollout_horizon):
            action = agent.act(obs, sample=sac_samples_action, batched=True)
            pred_next_obs, pred_rewards, pred_dones, model_state = dynamics_model.step(
                action, model_state, sample=True, model_updates=model_updates
            )
            if pred_rewards is None:
                # Model doesn't predict reward (learn_reward=False) -- compute it
                # analytically from the predicted next state instead.
                assert reward_fn is not None, (
                    "dynamics_model.step returned no reward but no reward_fn was given"
                )
                with torch.no_grad():
                    pred_rewards = reward_fn(
                        torch.from_numpy(action).float(),
                        torch.from_numpy(pred_next_obs).float(),
                    ).cpu().numpy()
            truncateds = np.zeros_like(pred_dones, dtype=bool)
            sac_buffer.add_batch(
                obs[~accum_dones],
                action[~accum_dones],
                pred_next_obs[~accum_dones],
                pred_rewards[~accum_dones, 0],
                pred_dones[~accum_dones, 0],
                truncateds[~accum_dones, 0],
            )
            obs = pred_next_obs
            accum_dones |= pred_dones.squeeze()

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
    reward_fn: Optional[types.TermFnType] = None,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)

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
    # If an analytic reward_fn was supplied (overrides.learned_rewards=false), the
    # model only predicts state deltas and reward is computed from the predicted
    # next state instead -- see rollout_model_and_populate_sac_buffer below. This
    # also keeps reward out of the forgetful_rl filtering entirely (it operates on
    # whatever the model actually predicts).
    learn_reward = reward_fn is None
    extra_model_kwargs = {}
    if str(cfg.dynamics_model._target_).endswith("GPModel"):
        # Only GPModel currently supports learn_reward=False; other dynamics models
        # (e.g. EnsembleModel) don't accept this kwarg, so don't pass it to them.
        extra_model_kwargs["learn_reward"] = learn_reward
    dynamics_model = hydra.utils.instantiate(
        cfg.dynamics_model,
        in_size=obs_shape[0] + act_shape[0],
        out_size=obs_shape[0] + (1 if learn_reward else 0),
        termination_fn=termination_fn,
        **extra_model_kwargs,
    )
    # Create optimizer only if model has parameters (for neural network models)
    # GP models handle their own optimization internally
    model_params = list(dynamics_model.parameters())
    model_optimizer = torch.optim.Adam(
        model_params, 
        lr=cfg.overrides.model_lr, 
        weight_decay=cfg.overrides.get("model_wd", 1e-5)
    ) if model_params else None
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
    rollout_total_batch_size = (
        cfg.overrides.effective_model_rollouts_per_step * cfg.algorithm.freq_train_model
    )
    rollout_batch_size = cfg.overrides.rollout_batch_size
    trains_per_epoch = int(
        np.ceil(cfg.overrides.epoch_length / cfg.overrides.freq_train_model)
    )
    updates_made = 0
    env_steps = 0
    best_eval_reward = -np.inf
    epoch = 0
    iteration = 0
    model_updates = 0

    rollout_length = int(
            utils.truncated_linear(
                *(cfg.overrides.rollout_schedule + [epoch + 1])
            )
        )
    sac_buffer_capacity = rollout_length * rollout_total_batch_size * trains_per_epoch
    sac_buffer_capacity *= cfg.overrides.num_epochs_to_retain_sac_buffer
    rng = np.random.default_rng(seed=cfg.seed)
    sac_buffer = ReplayBuffer(sac_buffer_capacity, obs_shape, act_shape, rng=rng)

    while env_steps < cfg.overrides.num_steps:
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
                env, obs, agent, {}, replay_buffer
            )

            # --------------- Model Training -----------------
            if (env_steps + 1) % cfg.overrides.freq_train_model == 0:
                avg_model_loss = dynamics_model.train_model(
                    replay_buffer=replay_buffer,
                    optimizer=model_optimizer,
                    batch_size=cfg.overrides.model_batch_size,
                    num_epochs=cfg.get("num_epochs_train_model", None),
                    patience=cfg.get("patience", 1),
                    improvement_threshold=cfg.get("improvement_threshold", 0.01),
                    model_lr=cfg.overrides.model_lr,
                    val_ratio=cfg.overrides.get("validation_ratio", 0.2),
                    model_wd=cfg.overrides.get("model_wd", 0.0),
                )
                model_updates += 1

                logger.log_data("results", {
                    "env_step": env_steps,
                    "train/model_loss": avg_model_loss,
                    "epoch": iteration,
                    "model_updates": model_updates
                })
                
                # --------------- Forgetful BO: Filter Replay Buffer -----------------
                if cfg.overrides.get("forgetful_rl", False) and len(sac_buffer) > 0:
                    try:
                        # Create target region from SAC buffer (model-generated states)
                        target_region = TargetRegion(
                            replay_buffer=sac_buffer,
                            env=env,
                            agent=agent,
                            num_initial_samples=cfg.overrides.get("forgetful_rl_num_initial_samples", 100),
                            seed=cfg.seed
                        )
                        
                        # Get target region statistics
                        tr_stats = target_region.get_statistics()
                        
                        # Filter the real environment replay buffer based on target region
                        # Only filter if we have a GP model (check for likelihood attribute)
                        if hasattr(dynamics_model, 'likelihood') and hasattr(dynamics_model, 'gp_model'):
                            if dynamics_model.gp_model is not None:
                                original_size = len(replay_buffer)
                                store_data_removal = cfg.overrides.get("store_data_removal", False)
                                filtered_buffer, filter_stats, removal_data = filter_replay_buffer(
                                    replay_buffer=replay_buffer,
                                    model=dynamics_model.gp_model,
                                    likelihood=dynamics_model.likelihood,
                                    target_region=target_region,
                                    max_iter=cfg.overrides.get("forgetful_rl_max_iter", 20),
                                    noise_type=cfg.overrides.get("forgetful_rl_noise_type", "joint"),
                                    max_samples=cfg.overrides.get("forgetful_rl_max_samples", 500),
                                    batch_optimize=cfg.overrides.get("forgetful_rl_batch_optimize", True),
                                    store_data_removal=store_data_removal,
                                )
                                
                                # Replace replay buffer with filtered version
                                replay_buffer = filtered_buffer
                                
                                # Compute statistics
                                filtered_size = len(filtered_buffer)
                                removed_samples = original_size - filtered_size
                                
                                # Log filtering statistics
                                logger.log_data("results", {
                                    "env_step": env_steps,
                                    "forgetful_rl/original_buffer_size": original_size,
                                    "forgetful_rl/filtered_buffer_size": filtered_size,
                                    "forgetful_rl/removed_samples": removed_samples,
                                    "forgetful_rl/removal_ratio": filter_stats['removal_ratio'],
                                    "forgetful_rl/target_region_size": tr_stats['target_region_size'],
                                    "forgetful_rl/target_region_ratio": tr_stats['target_region_ratio'],
                                    "forgetful_rl/threshold_value": tr_stats['threshold_value'],
                                })
                                # optionally log the actual sample data to wandb for visualization
                                if store_data_removal and removal_data is not None:
                                    # create a wandb table with kept/removed flags
                                    table = wandb.Table(columns=["obs", "action", "next_obs", "reward", "removed"])
                                    kept = removal_data['kept']
                                    removed = removal_data['removed']
                                    # helper to append rows
                                    def _add_rows(arr, removed_flag: bool):
                                        for i in range(len(arr['obs'])):
                                            table.add_data(
                                                arr['obs'][i].tolist(),
                                                arr['act'][i].tolist(),
                                                arr['next_obs'][i].tolist(),
                                                float(arr['rewards'][i]),
                                                removed_flag,
                                            )
                                    _add_rows(kept, False)
                                    _add_rows(removed, True)
                                    # log with current env_step so we can filter later
                                    wandb.log({
                                        "forgetful_rl/data_removal": table,
                                        "env_step": env_steps,
                                    })
                                
                                if debug_mode:
                                    print(f"\n[Forgetful BO] Filtered buffer: {original_size} -> {filtered_size} "
                                          f"(removed {filter_stats['removal_ratio']:.1%})")
                                    print(f"[Forgetful BO] Target region: {tr_stats['target_region_ratio']:.1%} of SAC buffer")
                        else:
                            # For ensemble models, just log target region stats
                            logger.log_data("results", {
                                "env_step": env_steps,
                                "forgetful_rl/target_region_size": tr_stats['target_region_size'],
                                "forgetful_rl/target_region_ratio": tr_stats['target_region_ratio'],
                                "forgetful_rl/threshold_value": tr_stats['threshold_value'],
                            })
                            
                    except Exception as e:
                        print(f"Warning: Forgetful BO filtering failed: {e}")
                        import traceback
                        traceback.print_exc()

                # --------- Rollout new model and store imagined trajectories --------
                # Batch all rollouts for the next freq_train_model steps together
                rollout_model_and_populate_sac_buffer(
                    dynamics_model,
                    replay_buffer,
                    agent,
                    sac_buffer,
                    cfg.algorithm.sac_samples_action,
                    rollout_length,
                    rollout_total_batch_size,
                    rollout_batch_size,
                    model_updates,
                    reward_fn=reward_fn,
                )

                if debug_mode:
                    print(
                        f"Epoch: {epoch}. "
                        f"SAC buffer size: {len(sac_buffer)}. "
                        f"Rollout length: {rollout_length}. "
                        f"Steps: {env_steps}"
                    )

            # --------------- Agent Training -----------------
            for _ in range(cfg.overrides.num_sac_updates_per_step):
                use_real_data = rng.random() < cfg.algorithm.real_data_ratio
                which_buffer = replay_buffer if use_real_data else sac_buffer
                if (env_steps + 1) % cfg.overrides.sac_updates_every_steps != 0 or len(
                    which_buffer
                ) < cfg.overrides.sac_batch_size:
                    break

                agent.sac_agent.update_parameters(
                    which_buffer,
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