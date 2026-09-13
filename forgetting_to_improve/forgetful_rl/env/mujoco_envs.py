# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Lazy imports to avoid mujoco_py dependency issues
# These will only be imported when actually used
def __getattr__(name):
    if name == "SparsePointRobot":
        from .sparse_point_robot import SparsePointRobot
        return SparsePointRobot
    elif name == "AntTruncatedObsEnv":
        from .ant_truncated_obs import AntTruncatedObsEnv
        return AntTruncatedObsEnv
    elif name == "HumanoidTruncatedObsEnv":
        from .humanoid_truncated_obs import HumanoidTruncatedObsEnv
        return HumanoidTruncatedObsEnv
    elif name == "MujocoGymPixelWrapper":
        from .mujoco_pixel_wrapper import MujocoGymPixelWrapper
        return MujocoGymPixelWrapper
    elif name == "CartPoleEnv":
        from .pets_cartpole import CartPoleEnv
        return CartPoleEnv
    elif name == "HalfCheetahEnv":
        from .pets_halfcheetah import HalfCheetahEnv
        return HalfCheetahEnv
    elif name == "PusherEnv":
        from .pets_pusher import PusherEnv
        return PusherEnv
    elif name == "Reacher3DEnv":
        from .pets_reacher import Reacher3DEnv
        return Reacher3DEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
