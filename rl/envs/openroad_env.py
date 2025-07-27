from __future__ import annotations
from typing import Any, Dict, Tuple

from .base_env import BaseEnv
from rl.runners.tool_runner import ToolRunner
from rl.runners.mock_runner import MockRunner

class OpenRoadEnv(BaseEnv):
    """
    An RL environment wrapping a single OpenROAD (or other tool) run.
    One step = choose params -> call tool -> parse QoR -> compute reward.
    For now, use MockRunner. Switch to ToolRunner when HW pipeline is ready.
    """

    def __init__(self,
                 design_name: str = "ac97_top",
                 work_dir: str = "/workspace",
                 use_mock: bool = True,
                 max_episode_len: int = 50):
        self.design_name = design_name
        self.work_dir = work_dir
        self.use_mock = use_mock
        self.max_episode_len = max_episode_len

        self.runner = MockRunner() if use_mock else ToolRunner(work_dir=work_dir)

        # Internal state
        self._last_obs: Dict[str, Any] = {}
        self._done = False
        self._step_count = 0

        # Very rough placeholders for spaces
        self._action_space = {
            "type": "dict",
            "keys": ["param1", "param2"],
            "ranges": [(0, 10), (0.0, 1.0)]
        }
        self._observation_space = {
            "wirelength": float,
            "timing_slack": float,
            "power": float
        }

    def reset(self) -> Dict[str, Any]:
        self._done = False
        self._step_count = 0
        # In a real setup, you could call a script that resets the design state
        obs = {"wirelength": 0.0, "timing_slack": 0.0, "power": 0.0}
        self._last_obs = obs
        return obs

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before stepping again.")

        self._step_count += 1

        # 1) Apply action -> run tool
        if self.use_mock:
            qor = self.runner.run(action)
        else:
            # TODO: Write a script that consumes `action` and calls OpenROAD with parameters.
            # Example placeholder:
            # out = self.runner.run_openroad_py("rl/scripts/run_openroad_with_params.py", extra_args=[...])
            # qor = parse_stdout_to_qor(out)
            qor = {"wirelength": 1.0, "timing_slack": 0.0, "power": 100.0}

        # 2) Compute reward
        reward = self._compute_reward(qor)

        # 3) Check termination condition
        done = self._step_count >= self.max_episode_len
        self._done = done

        info = {"design": self.design_name, "action": action, "step": self._step_count}
        self._last_obs = qor
        return qor, reward, done, info

    def _compute_reward(self, qor: Dict[str, float]) -> float:
        """
        Example reward: minimize wirelength and power, penalize negative slack.
        Tune the scale factors based on real value ranges later.
        """
        wl = qor["wirelength"]
        p = qor["power"]
        slack = qor["timing_slack"]
        reward = -1e-6 * wl - 0.1 * p
        if slack < 0:
            reward += slack  # more penalty
        return reward

    def action_space(self) -> Any:
        return self._action_space

    def observation_space(self) -> Any:
        return self._observation_space

    def close(self):
        # Clean up if necessary
        pass
