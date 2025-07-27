from __future__ import annotations
from typing import Dict, Any
import numpy as np
from collections import deque

class ReplayBuffer:
    """
    A very simple FIFO replay buffer for off-policy methods like DQN.
    """

    def __init__(self, capacity: int, obs_dim: int = 3):
        self.capacity = capacity
        self.obs_dim = obs_dim

        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add(self, obs_vec: np.ndarray, action: int, reward: float,
            next_obs_vec: np.ndarray, done: bool):
        idx = self.ptr
        self.obs_buf[idx] = obs_vec
        self.act_buf[idx] = action
        self.rew_buf[idx] = reward
        self.next_obs_buf[idx] = next_obs_vec
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": self.obs_buf[idxs],
            "actions": self.act_buf[idxs],
            "rewards": self.rew_buf[idxs],
            "next_obs": self.next_obs_buf[idxs],
            "dones": self.done_buf[idxs],
        }

    def __len__(self):
        return self.size
