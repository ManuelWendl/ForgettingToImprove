import numpy as np
import torch


class SACAgent:
    """Wraps a pytorch_sac_pranz24 SAC learner with an `act` method."""

    def __init__(self, sac_learner):
        self.sac_agent = sac_learner

    def act(self, obs: np.ndarray, sample: bool = False, batched: bool = False, **kwargs) -> np.ndarray:
        with torch.no_grad():
            return self.sac_agent.select_action(
                obs, batched=batched, evaluate=not sample
            )
