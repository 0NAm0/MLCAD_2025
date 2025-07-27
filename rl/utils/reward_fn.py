from __future__ import annotations
from typing import Dict, Any

class RewardFn:
    """
    Configurable reward calculator.
    Example config:
    reward:
      weights:
        wirelength: -1e-6
        power_dynamic: -0.1
        timing_slack: 1.0
      slack_penalty_if_negative: 1.0   # extra penalty factor
      normalize:
        wirelength: {ref: 1.5e6, scale: 1.0}
        power_dynamic: {ref: 150, scale: 1.0}
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.weights = cfg.get("weights", {})
        self.slack_penalty_if_negative = cfg.get("slack_penalty_if_negative", 0.0)
        self.norm_cfg = cfg.get("normalize", {})

    def __call__(self, qor: Dict[str, float]) -> float:
        score = 0.0
        for k, w in self.weights.items():
            if k not in qor:
                continue
            val = qor[k]
            # normalization
            if k in self.norm_cfg:
                ref = self.norm_cfg[k].get("ref", 1.0)
                scale = self.norm_cfg[k].get("scale", 1.0)
                val = (val - ref) / scale
            score += w * val

        # Special handling for negative slack if requested
        if "timing_slack" in qor and qor["timing_slack"] < 0:
            score += self.slack_penalty_if_negative * qor["timing_slack"]

        return float(score)
