from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

class BaseEnv(ABC):
    """
    A minimal RL environment interface similar to OpenAI Gym.
    Implement reset() and step() in concrete subclasses.
    """

    @abstractmethod
    def reset(self) -> Dict[str, Any]:
        """Reset environment and return the initial observation."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take an action and return:
          observation (dict), reward (float), done (bool), info (dict)
        """
        raise NotImplementedError

    @abstractmethod
    def action_space(self) -> Any:
        """Return a description/object of the action space."""
        raise NotImplementedError

    @abstractmethod
    def observation_space(self) -> Any:
        """Return a description/object of the observation space."""
        raise NotImplementedError

    def close(self):
        """Optional cleanup."""
        pass
