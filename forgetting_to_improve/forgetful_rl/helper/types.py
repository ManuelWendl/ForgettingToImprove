from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

TermFnType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
TensorType = Union[torch.Tensor, np.ndarray]

Transition = Tuple[
    TensorType, TensorType, TensorType, TensorType, TensorType, TensorType
]

@dataclass
class TransitionBatch:
    """Represents a batch of transitions"""

    obs: Optional[TensorType]
    act: Optional[TensorType]
    next_obs: Optional[TensorType]
    rewards: Optional[TensorType]
    terminateds: Optional[TensorType]
    truncateds: Optional[TensorType]

    def __len__(self):
        return self.obs.shape[0]

    def astuple(self) -> Transition:
        return (
            self.obs,
            self.act,
            self.next_obs,
            self.rewards,
            self.terminateds,
            self.truncateds,
        )

    def __getitem__(self, item):
        return TransitionBatch(
            self.obs[item],
            self.act[item],
            self.next_obs[item],
            self.rewards[item],
            self.terminateds[item],
            self.truncateds[item],
        )