from __future__ import annotations
from typing import Dict, Any
import numpy as np

from rl.envs.base_env import BaseEnv
from rl.agents.base_agent import BaseAgent
from rl.memory.replay_buffer import ReplayBuffer
from rl.utils.logger import Logger

class Trainer:
    """
    Generic training loop for off-policy RL (e.g., DQN).
    Assumes discrete actions and vectorized obs (adapt as needed).
    """

    def __init__(self,
                 env: BaseEnv,
                 agent: BaseAgent,
                 cfg: Dict[str, Any],
                 log: Logger | None = None):
        self.env = env
        self.agent = agent
        self.cfg = cfg
        self.total_steps = cfg.get("total_steps", 10000)
        self.max_episode_len = cfg.get("max_episode_len", 50)
        self.buffer = ReplayBuffer(cfg.get("replay_size", 50000))
        self.batch_size = cfg.get("batch_size", 64)
        self.log = log or Logger()

    def _obs_to_vec(self, obs: Dict[str, Any]) -> np.ndarray:
        # Adapt for real obs; keep consistent with agent._obs_to_vec
        return np.array([obs["wirelength"], obs["timing_slack"], obs["power"]], dtype=np.float32)

    def train(self):
        steps = 0
        while steps < self.total_steps:
            obs = self.env.reset()
            ep_reward = 0.0

            for t in range(self.max_episode_len):
                obs_vec = self._obs_to_vec(obs)
                action = self.agent.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                next_obs_vec = self._obs_to_vec(next_obs)

                self.buffer.add(obs_vec, action, reward, next_obs_vec, done)
                ep_reward += reward
                steps += 1

                # Learn when enough data collected
                if len(self.buffer) >= self.batch_size:
                    batch = self.buffer.sample(self.batch_size)
                    stats = self.agent.learn(batch)
                    self.log.log_dict(stats, step=steps)

                if done or steps >= self.total_steps:
                    break
                obs = next_obs

            self.log.log_scalar("episode_reward", ep_reward, step=steps)
        self.log.flush()
