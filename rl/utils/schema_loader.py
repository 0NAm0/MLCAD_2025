from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import yaml

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    with open(p, "r") as f:
        return yaml.safe_load(f)

def load_actions_schema(path: str | Path) -> Dict[str, Any]:
    data = load_yaml(path)
    if "actions" not in data:
        raise ValueError("actions.yaml must contain top-level key 'actions'")
    return data

def load_qor_schema(path: str | Path) -> Dict[str, Any]:
    data = load_yaml(path)
    if "metrics" not in data:
        raise ValueError("qor_schema.yaml must contain top-level key 'metrics'")
    return data
