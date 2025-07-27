from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class QNetwork(nn.Module):
    """
    Very small MLP for Q-value estimation.
    This is a placeholder; adapt input/output dims to your real obs/action encoding.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim)
        )

    def forward(self, x):
        return self.net(x)

class DQNAgent:
    """
    Minimal DQN agent (discrete action example).
    NOTE: If your action is continuous or a dict, this class must be adapted.
    """

    def __init__(self, cfg: Dict[str, Any], obs_dim: int = 3, act_dim: int = 11):
        """
        obs_dim: number of observation values
        act_dim: number of discrete actions (placeholder)
        """
        self.gamma = cfg.get("gamma", 0.99)
        self.lr = cfg.get("lr", 1e-4)
        self.batch_size = cfg.get("batch_size", 64)
        self.target_update_interval = cfg.get("target_update_interval", 1000)

        self.eps_start = cfg.get("epsilon_start", 1.0)
        self.eps_end = cfg.get("epsilon_end", 0.05)
        self.eps_decay_steps = cfg.get("epsilon_decay_steps", 8000)
        self.global_step = 0

        self.q_net = QNetwork(obs_dim, act_dim)
        self.target_q_net = QNetwork(obs_dim, act_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optim = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.act_dim = act_dim

    def _epsilon(self) -> float:
        """Linear decay of epsilon."""
        ratio = min(self.global_step / self.eps_decay_steps, 1.0)
        return self.eps_start + ratio * (self.eps_end - self.eps_start)

    def select_action(self, obs: Dict[str, Any]) -> int:
        """
        Epsilon-greedy action selection.
        Convert obs(dict) -> np array; this is a placeholder.
        """
        self.global_step += 1
        eps = self._epsilon()

        obs_vec = self._obs_to_vec(obs)
        if np.random.rand() < eps:
            return np.random.randint(self.act_dim)
        with torch.no_grad():
            q_values = self.q_net(torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0))
            return int(torch.argmax(q_values, dim=1).item())

    def learn(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        batch: dict with keys: obs, actions, rewards, next_obs, dones
        Each value is np.ndarray
        """
        obs = torch.tensor(batch["obs"], dtype=torch.float32)
        actions = torch.tensor(batch["actions"], dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32).unsqueeze(1)
        next_obs = torch.tensor(batch["next_obs"], dtype=torch.float32)
        dones = torch.tensor(batch["dones"], dtype=torch.float32).unsqueeze(1)

        # Q(s,a)
        q_values = self.q_net(obs).gather(1, actions)

        # Target Q
        with torch.no_grad():
            max_next_q = self.target_q_net(next_obs).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * max_next_q

        loss = self.loss_fn(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # Periodically sync target network
        if self.global_step % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return {"loss": float(loss.item())}

    def save(self, path: str):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path: str):
        self.q_net.load_state_dict(torch.load(path, map_location="cpu"))
        self.target_q_net.load_state_dict(self.q_net.state_dict())

    def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Convert observation dict to a flat vector.
        Adapt this for real obs structure.
        """
        return np.array([obs["wirelength"], obs["timing_slack"], obs["power"]], dtype=np.float32)
