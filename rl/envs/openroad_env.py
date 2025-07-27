from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

from .base_env import BaseEnv
from rl.runners.tool_runner import ToolRunner
from rl.runners.mock_runner import MockRunner
from rl.utils.schema_loader import load_actions_schema, load_qor_schema
from rl.utils.action_encoder import ActionEncoder
from rl.utils.reward_fn import RewardFn


class OpenRoadEnv(BaseEnv):
    """
    RL environment wrapping one OpenROAD (or other EDA tool) run per step.

    One step flow:
      1) Agent outputs a *discrete* action index.
      2) We decode it to a param dict via ActionEncoder.
      3) Call real tools (ToolRunner) or MockRunner to get QoR.
      4) Compute reward via RewardFn.
      5) Return (obs_dict=QoR, reward, done, info).

    Observation is the raw QoR dict for now. Adapt to vectorized obs outside if needed.
    """

    def __init__(self,
                 design_name: str = "ac97_top",
                 work_dir: str = "/workspace",
                 use_mock: bool = True,
                 max_episode_len: int = 50,
                 actions_schema: str = "rl/interfaces/actions.yaml",
                 qor_schema: str = "rl/interfaces/qor_schema.yaml",
                 reward_cfg: Optional[Dict[str, Any]] = None):
        # Basic config
        self.design_name = design_name
        self.work_dir = work_dir
        self.use_mock = use_mock
        self.max_episode_len = max_episode_len

        # Runner selection
        self.runner = MockRunner() if use_mock else ToolRunner(work_dir=work_dir)

        # Schemas & helpers
        self.actions_schema_path = actions_schema
        self.qor_schema_path = qor_schema

        action_schema = load_actions_schema(actions_schema)
        self.action_encoder = ActionEncoder(action_schema)

        # QoR schema currently unused except for validation/consistency checks
        self.qor_schema = load_qor_schema(qor_schema)

        self.reward_fn = RewardFn(reward_cfg or {})

        # Internal state
        self._last_obs: Dict[str, Any] = {}
        self._done = False
        self._step_count = 0

        # Spaces (lightweight descriptors)
        self._action_space = {"discrete": self.action_encoder.num_actions}
        # You can extend this with all metrics from qor_schema if desired
        self._observation_space = {"wirelength": float, "timing_slack": float, "power_dynamic": float}

    # --------------------------------------------------------------------- #
    # BaseEnv interface
    # --------------------------------------------------------------------- #
    def reset(self) -> Dict[str, Any]:
        """Reset episode state and return an initial dummy observation."""
        self._done = False
        self._step_count = 0
        obs = {"wirelength": 0.0, "timing_slack": 0.0, "power_dynamic": 0.0}
        self._last_obs = obs
        return obs

    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        action_idx: discrete action index produced by the agent (DQN etc.).
        Returns: (obs_dict, reward, done, info)
        """
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before stepping again.")

        self._step_count += 1

        # 1) Decode discrete index -> param dict
        action_dict = self.action_encoder.index_to_dict(action_idx)

        # 2) Run tool
        if self.use_mock:
            qor = self.runner.run(action_dict)
        else:
            qor = self.runner.run_openroad(action_dict, design=self.design_name)

        # 3) Compute reward
        reward = self.reward_fn(qor)

        # 4) Termination condition
        done = self._step_count >= self.max_episode_len
        self._done = done

        info = {
            "design": self.design_name,
            "action_dict": action_dict,
            "action_idx": action_idx,
            "step": self._step_count
        }

        self._last_obs = qor
        return qor, reward, done, info

    def action_space(self) -> Any:
        """Return a description of the (discrete) action space size."""
        return self._action_space

    def observation_space(self) -> Any:
        """Return a minimal description of observation fields."""
        return self._observation_space

    def close(self):
        """Optional cleanup."""
        pass
