# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np


def quantize_obs(
    obs: np.ndarray,
    bit_depth: int,
    original_bit_depth: int = 8,
    add_noise: bool = False,
):
    """Quantizes an array of pixel observations to the desired bit depth.

    Args:
        obs (np.ndarray): the array to quantize.
        bit_depth (int): the desired bit depth.
        original_bit_depth (int, optional): the original bit depth, defaults to 8.
        add_noise (bool, optional): if ``True``, uniform noise in the range
            (0, 2 ** (8 - bit_depth)) will be added. Defaults to ``False``.`

    Returns:
        (np.ndarray): the quantized version of the array.
    """
    ratio = 2 ** (original_bit_depth - bit_depth)
    quantized_obs = (obs // ratio) * ratio
    if add_noise:
        quantized_obs = quantized_obs.astype(np.double) + ratio * np.random.rand(
            *obs.shape
        )
    return quantized_obs
