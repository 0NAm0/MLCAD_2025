from __future__ import annotations
import random
from typing import Dict, Any

class MockRunner:
    """
    A simple stub to simulate QoR outputs before real tools are wired in.
    """

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def run(self, action: Any) -> Dict[str, float]:
        """
        Returns fake QoR metrics given an action.
        You can customize how the action affects the random values.
        """
        return {
            "wirelength": random.uniform(1.0e6, 2.0e6),
            "timing_slack": random.uniform(-0.2, 0.1),
            "power": random.uniform(90.0, 200.0)
        }
