# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import omegaconf
import torch
import wandb
from omegaconf import OmegaConf

from .algorithm import sac, mbpo
from .helper.env import EnvHandler

import os
os.environ["WANDB_DISABLE_GPU"] = "true"
os.environ["WANDB_DISABLE_HARDWARE"] = "true"

@hydra.main(config_path="config", config_name="main")
def run(cfg: omegaconf.DictConfig):

    # Initialize wandb run
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    exp_name = f"{cfg.overrides.exp_name}_seed_{cfg.seed}"
    wandb.init(name=exp_name, project=cfg.overrides.project_name, entity="manuelwendl-eth-z-rich", config=cfg_dict, reinit=True)

    # Train the experiment
    print(f"Training with seed {cfg.seed}")
    env, term_fn, reward_fn = EnvHandler.make_env(cfg)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "sac":
        test_env, *_ = EnvHandler.make_env(cfg)
        return sac.train(env, test_env, term_fn, cfg)
    # if cfg.algorithm.name == "pets":
    #     return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = EnvHandler.make_env(cfg)
        return mbpo.train(env, test_env, term_fn, cfg, reward_fn=reward_fn)


if __name__ == "__main__":
    run()