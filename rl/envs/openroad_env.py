"""
openroad_env.py
RL environment that wraps a single OpenROAD (or mock) run per step.

"""

from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

from pathlib import Path
import importlib

from .base_env import BaseEnv
from rl.runners.tool_runner import ToolRunner
from rl.runners.mock_runner import MockRunner
from rl.utils.schema_loader import load_actions_schema, load_qor_schema
from rl.utils.action_encoder import ActionEncoder
from rl.utils.reward_fn import RewardFn

# helper import
from rl.envs.openroad_helpers import load_design, get_qor_metrics


class OpenRoadEnv(BaseEnv):
    """
    One RL step = choose parameters (or netlist transformation) → run tool
    → get QoR → compute reward.

    Observation today = QoR dict (wns / tns / power / wl / congestion).
    You can append more structural features later.
    """

    # ctor
    def __init__(
        self,
        design_name: str = "ac97_top",
        work_dir: str = "/workspace",
        use_mock: bool = True,
        max_episode_len: int = 50,
        actions_schema: str = "rl/interfaces/actions.yaml",
        qor_schema: str = "rl/interfaces/qor_schema.yaml",
        reward_cfg: Optional[Dict[str, Any]] = None,
    ):
        # basic
        self.design_name = design_name
        self.work_dir = work_dir
        self.use_mock = use_mock
        self.max_episode_len = max_episode_len

        # runner
        self.runner = MockRunner() if use_mock else ToolRunner(work_dir=work_dir)

        # schemas / helpers
        self.action_encoder = ActionEncoder(load_actions_schema(actions_schema))
        self.qor_schema = load_qor_schema(qor_schema)  # unused yet, but keeps consistency
        self.reward_fn = RewardFn(reward_cfg or {})

        # state
        self._done = False
        self._step_count = 0
        self._last_obs: Dict[str, Any] = {}

        # simple descriptors (can expand)
        self._action_space = {"discrete": self.action_encoder.num_actions}
        self._observation_space = {
            "wns": float,
            "tns": float,
            "power": float,
            "wirelength": float,
            "congestion": float,
        }

    # -----------BaseEnv impl--------------
    def reset(self) -> Dict[str, Any]:
        self._done = False
        self._step_count = 0
        obs = dict.fromkeys(self._observation_space.keys(), 0.0)
        self._last_obs = obs
        return obs

    # main step
    def step(self, action_idx: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode ended, call reset()")

        self._step_count += 1

        # idx → param dict
        action_dict = self.action_encoder.index_to_dict(action_idx)

        # Branch 1 : mock
        if self.use_mock:
            qor = self.runner.run(action_dict)

        # Branch 2 : real flow
        else:
            # 1) run the full flow (params.json already dumped by ToolRunner)
            self.runner.run_openroad(action_dict, design=self.design_name)

            # 2) reopen OpenROAD design DB to pull QoR
            import openroad  # inside container image
            design_dir = Path(self.work_dir) / "designs" / self.design_name / "EDA_files"
            tech_json   = design_dir / f"{self.design_name}_tech.json"
            design_json = design_dir / f"{self.design_name}.json"

            if not tech_json.exists() or not design_json.exists():
                raise FileNotFoundError(
                    f"Design DB not found: {design_json} / {tech_json}"
                )

            _, design_db = load_design(tech_json, design_json, openroad)
            qor = get_qor_metrics(design_db)

        # reward & done
        reward = self.reward_fn(qor)
        done = self._step_count >= self.max_episode_len
        self._done = done

        info = {
            "step": self._step_count,
            "design": self.design_name,
            "action_idx": action_idx,
            "action_dict": action_dict,
        }

        self._last_obs = qor
        return qor, reward, done, info

    # simple descriptors
    def action_space(self) -> Any:
        return self._action_space

    def observation_space(self) -> Any:
        return self._observation_space

    def close(self):
        pass
