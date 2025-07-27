from __future__ import annotations
from typing import Dict, Any
import json
import os

class Logger:
    """
    Minimalistic logger that prints to stdout and optionally saves JSON lines.
    """

    def __init__(self, out_dir: str = "./rl_logs", filename: str = "log.jsonl"):
        self.out_dir = out_dir
        self.filepath = os.path.join(out_dir, filename)
        os.makedirs(out_dir, exist_ok=True)
        self._fp = open(self.filepath, "a", encoding="utf-8")

    def log_scalar(self, name: str, value: float, step: int):
        record = {"step": step, name: value}
        print(f"[{step}] {name}: {value}")
        self._fp.write(json.dumps(record) + "\n")

    def log_dict(self, d: Dict[str, Any], step: int):
        record = {"step": step}
        record.update(d)
        print(f"[{step}] {d}")
        self._fp.write(json.dumps(record) + "\n")

    def flush(self):
        self._fp.flush()

    def close(self):
        self._fp.close()
