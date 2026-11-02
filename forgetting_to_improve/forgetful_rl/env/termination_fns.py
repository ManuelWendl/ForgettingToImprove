# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math

import torch


def cartpole(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    assert len(next_obs.shape) == 2

    x, theta = next_obs[:, 0], next_obs[:, 2]

    x_threshold = 2.4
    theta_threshold_radians = 12 * 2 * math.pi / 360
    not_done = (
        (x > -x_threshold)
        * (x < x_threshold)
        * (theta > -theta_threshold_radians)
        * (theta < theta_threshold_radians)
    )
    done = ~not_done
    done = done[:, None]
    return done


def no_termination(act: torch.Tensor, next_obs: torch.Tensor) -> torch.Tensor:
    import numpy as np
    
    # Convert to torch tensor if numpy array
    was_numpy = isinstance(next_obs, np.ndarray)
    if was_numpy:
        next_obs = torch.from_numpy(next_obs)
    
    assert len(next_obs.shape) == 2

    done = torch.Tensor([False]).repeat(len(next_obs)).bool().to(next_obs.device)
    done = done[:, None]
    
    # Return numpy array if input was numpy
    if was_numpy:
        return done.numpy()
    return done
