import pathlib
import warnings

from typing import List, Optional, Sequence, Tuple, Type, Union

from ..helper.types import TransitionBatch

import numpy as np

class ReplayBuffer:
    """A replay buffer with support for training/validation iterators and ensembles.

    This buffer can be pushed to and sampled from as a typical replay buffer.

    Args:
        capacity (int): the maximum number of transitions that the buffer can store.
            When the capacity is reached, the contents are overwritten in FIFO fashion.
        obs_shape (Sequence of ints): the shape of the observations to store.
        action_shape (Sequence of ints): the shape of the actions to store.
        obs_type (type): the data type of the observations (defaults to np.float32).
        action_type (type): the data type of the actions (defaults to np.float32).
        reward_type (type): the data type of the rewards (defaults to np.float32).
        rng (np.random.Generator, optional): a random number generator when sampling
            batches. If None (default value), a new default generator will be used.
        max_trajectory_length (int, optional): if given, indicates that trajectory
            information should be stored and that trajectories will be at most this
            number of steps. Defaults to ``None`` in which case no trajectory
            information will be kept. The buffer will keep trajectory information
            automatically using the terminated value when calling :meth:`add`.

    .. warning::
        When using ``max_trajectory_length`` it is the user's responsibility to ensure
        that trajectories are stored continuously in the replay buffer.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Sequence[int],
        action_shape: Sequence[int],
        obs_type: Type = np.float32,
        action_type: Type = np.float32,
        reward_type: Type = np.float32,
        rng: Optional[np.random.Generator] = None,
        max_trajectory_length: Optional[int] = None,
    ):
        self.cur_idx = 0
        self.capacity = capacity
        self.num_stored = 0

        self.trajectory_indices: Optional[List[Tuple[int, int]]] = None
        if max_trajectory_length:
            self.trajectory_indices = []
            capacity += max_trajectory_length
        self.obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=obs_type)
        self.action = np.empty((capacity, *action_shape), dtype=action_type)
        self.reward = np.empty(capacity, dtype=reward_type)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)

        if rng is None:
            self._rng = np.random.default_rng()
        else:
            self._rng = rng

        self._start_last_trajectory = 0

    @property
    def stores_trajectories(self) -> bool:
        return self.trajectory_indices is not None

    @staticmethod
    def _check_overlap(segment1: Tuple[int, int], segment2: Tuple[int, int]) -> bool:
        s1, e1 = segment1
        s2, e2 = segment2
        return (s1 <= s2 < e1) or (s1 < e2 <= e1)

    def remove_overlapping_trajectories(self, new_trajectory: Tuple[int, int]):
        cnt = 0
        for traj in self.trajectory_indices:
            if self._check_overlap(new_trajectory, traj):
                cnt += 1
            else:
                break
        for _ in range(cnt):
            self.trajectory_indices.pop(0)

    def _trajectory_bookkeeping(self, terminated: bool):
        self.cur_idx += 1
        if self.num_stored < self.capacity:
            self.num_stored += 1
        if self.cur_idx >= self.capacity:
            self.num_stored = max(self.num_stored, self.cur_idx)
        if terminated:
            self.close_trajectory()
        else:
            partial_trajectory = (self._start_last_trajectory, self.cur_idx + 1)
            self.remove_overlapping_trajectories(partial_trajectory)
        if self.cur_idx >= len(self.obs):
            warnings.warn(
                "The replay buffer was filled before current trajectory finished. "
                "The history of the current partial trajectory will be discarded. "
                "Make sure you set `max_trajectory_length` to the appropriate value "
                "for your problem."
            )
            self._start_last_trajectory = 0
            self.cur_idx = 0
            self.num_stored = len(self.obs)

    def close_trajectory(self):
        new_trajectory = (self._start_last_trajectory, self.cur_idx)
        self.remove_overlapping_trajectories(new_trajectory)
        self.trajectory_indices.append(new_trajectory)

        if self.cur_idx - self._start_last_trajectory > (len(self.obs) - self.capacity):
            warnings.warn(
                "A trajectory was saved with length longer than expected. "
                "Unexpected behavior might occur."
            )

        if self.cur_idx >= self.capacity:
            self.cur_idx = 0
        self._start_last_trajectory = self.cur_idx

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: float,
        terminated: bool,
        truncated: bool,
    ):
        """Adds a transition (s, a, s', r, terminated) to the replay buffer.

        Args:
            obs (np.ndarray): the observation at time t.
            action (np.ndarray): the action at time t.
            next_obs (np.ndarray): the observation at time t + 1.
            reward (float): the reward at time t + 1.
            terminated (bool): a boolean indicating whether the episode ended in a terminal state.
            truncated (bool): a boolean indicating whether the episode ended prematurely.
        """
        self.obs[self.cur_idx] = obs
        self.next_obs[self.cur_idx] = next_obs
        self.action[self.cur_idx] = action
        self.reward[self.cur_idx] = reward
        self.terminated[self.cur_idx] = terminated
        self.truncated[self.cur_idx] = truncated

        if self.trajectory_indices is not None:
            self._trajectory_bookkeeping(terminated or truncated)
        else:
            self.cur_idx = (self.cur_idx + 1) % self.capacity
            self.num_stored = min(self.num_stored + 1, self.capacity)

    def add_batch(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        next_obs: np.ndarray,
        reward: np.ndarray,
        terminated: np.ndarray,
        truncated: np.ndarray,
    ):
        """Adds a transition (s, a, s', r, terminated, truncated) to the replay buffer.

        Expected shapes are:
            obs --> (batch_size,) + obs_shape
            act --> (batch_size,) + action_shape
            reward/terminated/truncated --> (batch_size,)

        Args:
            obs (np.ndarray): the batch of observations at time t.
            action (np.ndarray): the batch of actions at time t.
            next_obs (np.ndarray): the batch of observations at time t + 1.
            reward (float): the batch of rewards at time t + 1.
            terminated (bool): a batch of booleans terminal indicators.
            truncated (bool): a batch of booleans truncation indicators.
        """

        def copy_from_to(buffer_start, batch_start, how_many):
            buffer_slice = slice(buffer_start, buffer_start + how_many)
            batch_slice = slice(batch_start, batch_start + how_many)
            np.copyto(self.obs[buffer_slice], obs[batch_slice])
            np.copyto(self.action[buffer_slice], action[batch_slice])
            np.copyto(self.reward[buffer_slice], reward[batch_slice])
            np.copyto(self.next_obs[buffer_slice], next_obs[batch_slice])
            np.copyto(self.terminated[buffer_slice], terminated[batch_slice])
            np.copyto(self.truncated[buffer_slice], truncated[batch_slice])

        _batch_start = 0
        buffer_end = self.cur_idx + len(obs)
        if buffer_end > self.capacity:
            copy_from_to(self.cur_idx, _batch_start, self.capacity - self.cur_idx)
            _batch_start = self.capacity - self.cur_idx
            self.cur_idx = 0
            self.num_stored = self.capacity

        _how_many = len(obs) - _batch_start
        copy_from_to(self.cur_idx, _batch_start, _how_many)
        self.cur_idx = (self.cur_idx + _how_many) % self.capacity
        self.num_stored = min(self.num_stored + _how_many, self.capacity)

    def sample(self, batch_size: int) -> TransitionBatch:
        """Samples a batch of transitions from the replay buffer.

        Args:
            batch_size (int): the number of samples required.

        Returns:
            (tuple): the sampled values of observations, actions, next observations, rewards,
            terminated, and truncated indicators, as numpy arrays, respectively.
            The i-th transition corresponds to
            (obs[i], act[i], next_obs[i], rewards[i], terminateds[i], truncateds[i]).
        """
        indices = self._rng.choice(self.num_stored, size=batch_size)
        return self._batch_from_indices(indices)

    def sample_trajectory(self) -> Optional[TransitionBatch]:
        """Samples a full trajectory and returns it as a batch.

        Returns:
            (tuple): A tuple with observations, actions, next observations, rewards, terminated,
            and truncated indicators, as numpy arrays, respectively; these will correspond
            to a full trajectory. The i-th transition corresponds
            to (obs[i], act[i], next_obs[i], rewards[i], terminateds[i], truncateds[i]).
        """
        if self.trajectory_indices is None or len(self.trajectory_indices) == 0:
            return None
        idx = self._rng.choice(len(self.trajectory_indices))
        indices = np.arange(
            self.trajectory_indices[idx][0], self.trajectory_indices[idx][1]
        )
        return self._batch_from_indices(indices)

    def _batch_from_indices(self, indices) -> TransitionBatch:
        obs = self.obs[indices]
        next_obs = self.next_obs[indices]
        action = self.action[indices]
        reward = self.reward[indices]
        terminated = self.terminated[indices]
        truncated = self.truncated[indices]

        return TransitionBatch(obs, action, next_obs, reward, terminated, truncated)

    def __len__(self):
        return self.num_stored

    def save(self, save_dir: Union[pathlib.Path, str]):
        """Saves the data in the replay buffer to a given directory.

        Args:
            save_dir (str): the directory to save the data to. File name will be
                replay_buffer.npz.
        """
        path = pathlib.Path(save_dir) / "replay_buffer.npz"
        np.savez(
            path,
            obs=self.obs[: self.num_stored],
            next_obs=self.next_obs[: self.num_stored],
            action=self.action[: self.num_stored],
            reward=self.reward[: self.num_stored],
            terminated=self.terminated[: self.num_stored],
            truncated=self.truncated[: self.num_stored],
            trajectory_indices=self.trajectory_indices or [],
        )

    def load(self, load_dir: Union[pathlib.Path, str]):
        """Loads transition data from a given directory.

        Args:
            load_dir (str): the directory where the buffer is stored.
        """
        path = pathlib.Path(load_dir) / "replay_buffer.npz"
        data = np.load(path)
        num_stored = len(data["obs"])
        self.obs[:num_stored] = data["obs"]
        self.next_obs[:num_stored] = data["next_obs"]
        self.action[:num_stored] = data["action"]
        self.reward[:num_stored] = data["reward"]
        self.terminated[:num_stored] = data["terminated"]
        self.truncated[:num_stored] = data["truncated"]
        self.num_stored = num_stored
        self.cur_idx = self.num_stored % self.capacity
        if "trajectory_indices" in data and len(data["trajectory_indices"]):
            self.trajectory_indices = data["trajectory_indices"]

    def get_all(self, shuffle: bool = False) -> TransitionBatch:
        """Returns all data stored in the replay buffer.

        Args:
            shuffle (int): set to ``True`` if the data returned should be in random order.
            Defaults to ``False``.
        """
        if shuffle:
            permutation = self._rng.permutation(self.num_stored)
            return self._batch_from_indices(permutation)
        else:
            return TransitionBatch(
                self.obs[: self.num_stored],
                self.action[: self.num_stored],
                self.next_obs[: self.num_stored],
                self.reward[: self.num_stored],
                self.terminated[: self.num_stored],
                self.truncated[: self.num_stored],
            )

    @property
    def rng(self) -> np.random.Generator:
        return self._rng