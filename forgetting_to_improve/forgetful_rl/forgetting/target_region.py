import numpy as np
import torch
import gymnasium as gym
from typing import Optional

from ..data.buffer import ReplayBuffer
from ..algorithm.sac import SACAgent


class TargetRegion:
    """
    Defines a target region based on the agent's value function.
    
    The target region consists of all states in the replay buffer for which 
    the agent's value function (Q-value) is larger than the worst value 
    function of samples from the initial state distribution of the environment.
    """
    
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        env: gym.Env,
        agent: SACAgent,
        num_initial_samples: int = 100,
        seed: Optional[int] = None,
    ):
        """
        Initialize the target region.
        
        Args:
            replay_buffer: The SAC replay buffer containing state-action pairs
            env: The environment to sample initial states from
            agent: The trained SAC agent with critic network
            num_initial_samples: Number of initial state samples to determine threshold
            seed: Random seed for reproducible initial state sampling
        """
        self.replay_buffer = replay_buffer
        self.env = env
        self.agent = agent
        self.num_initial_samples = num_initial_samples
        self.seed = seed
        
        # Compute the threshold value (worst value from initial state distribution)
        self.threshold_value = self._compute_threshold()
        
        # Identify states in target region
        self.target_region_mask = self._compute_target_region_mask()
        
    def _compute_threshold(self) -> float:
        """
        Compute the threshold value as the minimum value function 
        over samples from the initial state distribution.
        
        Returns:
            The minimum Q-value from initial state samples
        """
        initial_states = []
        
        # Sample initial states from environment
        for i in range(self.num_initial_samples):
            seed = None if self.seed is None else self.seed + i
            state, _ = self.env.reset(seed=seed)
            initial_states.append(state)
        
        initial_states = np.array(initial_states)
        
        # Get Q-values for initial states
        # For each state, sample an action from the policy and compute Q-value
        q_values = []
        
        with torch.no_grad():
            states_tensor = torch.FloatTensor(initial_states).to(
                self.agent.sac_agent.device
            )
            
            # Sample actions from policy for each state
            actions, _, _ = self.agent.sac_agent.policy.sample(states_tensor)
            
            # Get Q-values from critic (take minimum of Q1 and Q2)
            q1, q2 = self.agent.sac_agent.critic(states_tensor, actions)
            q_values = torch.min(q1, q2).cpu().numpy().flatten()
        
        # Return the minimum (worst) Q-value
        threshold = np.min(q_values)
        
        return threshold
    
    def _compute_target_region_mask(self) -> np.ndarray:
        """
        Compute a boolean mask indicating which states in the replay buffer
        are in the target region (i.e., have value > threshold).
        
        Returns:
            Boolean array of shape (buffer_size,) indicating target region membership
        """
        # Get all states and actions from replay buffer
        all_data = self.replay_buffer.get_all()
        states = all_data.obs
        actions = all_data.act
        
        # Compute Q-values for all state-action pairs
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(
                self.agent.sac_agent.device
            )
            actions_tensor = torch.FloatTensor(actions).to(
                self.agent.sac_agent.device
            )
            
            # Get Q-values from critic (take minimum of Q1 and Q2)
            q1, q2 = self.agent.sac_agent.critic(states_tensor, actions_tensor)
            q_values = torch.max(q1, q2).cpu().numpy().flatten()

            actions_from_policy, _, _ = self.agent.sac_agent.policy.sample(states_tensor)
            q1_policy, q2_policy = self.agent.sac_agent.critic(states_tensor, actions_from_policy)
            q_values_policy = torch.min(q1_policy, q2_policy).cpu().numpy().flatten()
        
        # Create mask: True where Q-value > threshold
        mask = (q_values >= self.threshold_value) & (q_values >= q_values_policy)
        
        return mask
    
    def get_target_region_states(self) -> np.ndarray:
        """
        Get all states in the target region.
        
        Returns:
            Array of states that are in the target region
        """
        all_data = self.replay_buffer.get_all()
        return all_data.obs[self.target_region_mask]
    
    def get_target_region_ratio(self) -> float:
        """
        Get the fraction of replay buffer that is in the target region.
        
        Returns:
            Ratio of target region size to total buffer size
        """
        return np.mean(self.target_region_mask)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the target region.
        
        Returns:
            Dictionary with target region statistics
        """
        all_data = self.replay_buffer.get_all()
        target_states = all_data.obs[self.target_region_mask]
        
        return {
            "threshold_value": self.threshold_value,
            "target_region_size": int(np.sum(self.target_region_mask)),
            "total_buffer_size": len(all_data),
            "target_region_ratio": self.get_target_region_ratio(),
            "num_initial_samples": self.num_initial_samples,
        }