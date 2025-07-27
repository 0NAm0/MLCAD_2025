from __future__ import annotations
import subprocess
import os
from typing import Dict, Any, List, Optional

class ToolRunner:
    """
    Wrap calls to OpenROAD / GLOAM / CircuitOps (real EDA tools).
    Replace/extend run_* methods once the HW team finalizes their APIs.
    """

    def __init__(self,
                 openroad_bin: str = "/opt/mlcad/MLCAD25-Contest-Scripts-Benchmarks/OpenROAD/build/src/openroad",
                 work_dir: str = "/workspace",
                 env: Optional[Dict[str, str]] = None):
        self.openroad_bin = openroad_bin
        self.work_dir = work_dir
        self.env = env or os.environ.copy()

    def run_openroad_py(self, script_path: str, extra_args: Optional[List[str]] = None) -> str:
        """
        Run an OpenROAD python script:
          openroad -python script.py [extra_args...]
        Returns stdout text (to be parsed for QoR).
        """
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

    # TODO: add run_gloam(), run_circuitops() etc. once those binaries/paths are finalized.
