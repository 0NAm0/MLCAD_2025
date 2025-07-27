from __future__ import annotations
import yaml
from typing import Dict, Any

def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)
