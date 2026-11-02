# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Modified for Forgetful RL: MBPO with a GP dynamics model whose real-data
# replay buffer is filtered with Forgetting to Improve.
import os

import hydra
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import OmegaConf

from .algorithm import mbpo
from .helper.env import EnvHandler

os.environ["WANDB_DISABLE_GPU"] = "true"
os.environ["WANDB_DISABLE_HARDWARE"] = "true"


@hydra.main(config_path="config", config_name="main", version_base="1.1")
def run(cfg: omegaconf.DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    exp_name = f"{cfg.overrides.exp_name}_seed_{cfg.seed}"
    wandb.init(
        name=exp_name,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        config=cfg_dict,
        reinit=True,
    )

    print(f"Training with seed {cfg.seed}")
    env, term_fn, reward_fn = EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name != "mbpo":
        raise ValueError(f"Unsupported algorithm: {cfg.algorithm.name}")
    test_env, *_ = EnvHandler.make_env(cfg)
    return mbpo.train(env, test_env, term_fn, cfg, reward_fn=reward_fn)


if __name__ == "__main__":
    run()
