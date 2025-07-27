from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    """
    Base class for agents. Implement select_action() and learn() at minimum.
    """

    @abstractmethod
    def select_action(self, obs: Dict[str, Any]) -> Any:
        """Given observation, return an action."""
        raise NotImplementedError

    @abstractmethod
    def learn(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Update agent parameters from a batch of experience.
        Return training stats (losses, etc.) for logging.
        """
        raise NotImplementedError

    def save(self, path: str):
        """Optional: save model parameters."""
        pass

    def load(self, path: str):
        """Optional: load model parameters."""
        pass
