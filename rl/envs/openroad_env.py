"""
openroad_env.py
Gym-style environment that wraps one OpenROAD (or mock) run per RL step.
Now supports **gate-level actions** (clone / split / etc.) in addition to
flow-parameter tuning.

Dependencies:
    • rl.runners.tool_runner.ToolRunner   – launches the full Tcl flow
    • rl.runners.hw_api.apply_gate_op     – applies a gate-level transform
    • rl.envs.openroad_helpers.{load_design,get_qor_metrics}
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Optional

from .base_env import BaseEnv
from rl.runners.tool_runner import ToolRunner
from rl.runners.mock_runner import MockRunner
from rl.runners.hw_api import apply_gate_op           # NEW
from rl.utils.schema_loader import load_actions_schema, load_qor_schema
from rl.utils.action_encoder import ActionEncoder
from rl.utils.reward_fn import RewardFn
from rl.envs.openroad_helpers import load_design, get_qor_metrics


class OpenRoadEnv(BaseEnv):
    # ---------------------------------------------------------------
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
        self.design_name = design_name
        self.work_dir = work_dir
        self.use_mock = use_mock
        self.max_episode_len = max_episode_len

        self.runner = MockRunner() if use_mock else ToolRunner(work_dir=work_dir)

        self.action_encoder = ActionEncoder(load_actions_schema(actions_schema))
        self.qor_schema = load_qor_schema(qor_schema)
        self.reward_fn = RewardFn(reward_cfg or {})

        self._done = False
        self._step_count = 0
        self._last_obs: Dict[str, Any] = {}

        self._action_space = {"discrete": self.action_encoder.num_actions}
        self._observation_space = {
            "wns": float,
            "tns": float,
            "power": float,
            "wirelength": float,
            "congestion": float,
        }

    # ---------------------------------------------------------------
    def _design_dir(self) -> Path:
        return Path(self.work_dir) / "designs" / self.design_name / "EDA_files"

    # ---------------------------------------------------------------
    def _collect_instance_names(self, master: str) -> list[str]:
        """Return list of instance names whose master cell == target_master."""
        import openroad
        tech, design = load_design(
            self._design_dir() / f"{self.design_name}_tech.json",
            self._design_dir() / f"{self.design_name}.json",
            openroad,
        )
        block = design.getBlock()
        return [inst.getName() for inst in block.getInsts()
                if inst.getMaster().getConstName() == master]

    # ---------------------------------------------------------------
    def reset(self) -> Dict[str, Any]:
        self._done = False
        self._step_count = 0

        # build gate-op table once we know available instances (mock path skips)
        if not self.use_mock and self.action_encoder.gate_op_specs:
            # assume first spec's target_master is representative; or iterate
            inst_list = []
            for spec in self.action_encoder.gate_op_specs:
                inst_list += self._collect_instance_names(spec["target_master"])
            self.action_encoder.build_gate_op_table(inst_list)
            self._action_space["discrete"] = self.action_encoder.num_actions

        obs = dict.fromkeys(self._observation_space.keys(), 0.0)
        self._last_obs = obs
        return obs

    # ---------------------------------------------------------------
    def step(
        self, action_idx: int
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Episode ended; call reset()")

        self._step_count += 1
        action_dict = self.action_encoder.index_to_dict(action_idx)

        # MOCK
        if self.use_mock:
            qor = self.runner.run(action_dict)

        # REAL
        else:
            # 1) full flow with chosen parameters
            self.runner.run_openroad(action_dict, design=self.design_name)

            # 2) reopen DB
            import openroad
            design_dir = self._design_dir()
            tech_json   = design_dir / f"{self.design_name}_tech.json"
            design_json = design_dir / f"{self.design_name}.json"
            _, design_db = load_design(tech_json, design_json, openroad)

            # 3) optional gate-level transform
            if action_dict["type"] == "gate_op":
                apply_gate_op(design_db, action_dict)
                # simple incremental fixes; adjust as needed
                design_db.getOpenRoad().detailedPlacement()
                design_db.getOpenRoad().routeDetail()

            # 4) extract QoR
            qor = get_qor_metrics(design_db)

        # -reward & bookkeeping 
        reward = self.reward_fn(qor)
        done = self._step_count >= self.max_episode_len
        self._done = done

        info = {
            "step": self._step_count,
            "action_idx": action_idx,
            "action_dict": action_dict,
            "design": self.design_name,
        }
        self._last_obs = qor
        return qor, reward, done, info

    # ------
    def action_space(self) -> Any:
        return self._action_space

    def observation_space(self) -> Any:
        return self._observation_space

    def close(self):
        pass
