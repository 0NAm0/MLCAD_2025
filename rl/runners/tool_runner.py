from __future__ import annotations
import subprocess
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from rl.utils.qor_parser import parse_stdout, load_qor_json

class ToolRunner:
    """
    Wrap calls to OpenROAD / other tools.
    """

    def __init__(self,
                 openroad_bin: str = "/opt/mlcad/MLCAD25-Contest-Scripts-Benchmarks/OpenROAD/build/src/openroad",
                 work_dir: str = "/workspace",
                 env: Optional[Dict[str, str]] = None):
        self.openroad_bin = openroad_bin
        self.work_dir = work_dir
        self.env = env or os.environ.copy()
        self._tmp_dir = Path(work_dir) / "tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def run_openroad_py(self, script_path: str, extra_args: Optional[List[str]] = None) -> str:
        cmd = [self.openroad_bin, "-python", script_path]
        if extra_args:
            cmd.extend(extra_args)
        result = subprocess.run(
            cmd,
            cwd=self.work_dir,
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False
        )
        return result.stdout

    # NEW
    def run_openroad(self, params: Dict[str, Any], design: str = "ac97_top") -> Dict[str, float]:
        """
        High-level API:
        1) dump params to JSON
        2) call run_openroad_with_params.py
        3) parse stdout or JSON for QoR
        """
        params_path = self._tmp_dir / "params.json"
        with open(params_path, "w") as f:
            json.dump(params, f)

        out_qor_path = self._tmp_dir / "qor.json"
        script_rel = "rl/scripts/run_openroad_with_params.py"  # relative to /workspace
        extra_args = ["--params", str(params_path),
                      "--design", design,
                      "--out_qor", str(out_qor_path)]

        stdout = self.run_openroad_py(script_rel, extra_args)

        # Prefer JSON if it exists; fallback to stdout parse
        if out_qor_path.exists():
            return load_qor_json(out_qor_path)
        else:
            return parse_stdout(stdout)
