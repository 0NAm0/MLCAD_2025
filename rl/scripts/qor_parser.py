"""
Utilities to parse QoR metrics from tool outputs.

Two common entry points:
- parse_stdout(text): parse "key=value" pairs from stdout.
- load_qor_json(path): read a JSON QoR dump.

Extend as needed once HW side stabilizes the output format.
"""

from __future__ import annotations
from typing import Dict
import json
import re
from pathlib import Path

_KEYVAL_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*([-+Ee0-9\.]+)\s*$")

def parse_stdout(stdout: str) -> Dict[str, float]:
    qor: Dict[str, float] = {}
    for line in stdout.splitlines():
        m = _KEYVAL_RE.match(line)
        if m:
            key, val = m.group(1), m.group(2)
            try:
                qor[key] = float(val)
            except ValueError:
                # keep string if not numeric
                pass
    return qor

def load_qor_json(path: str | Path) -> Dict[str, float]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"QoR JSON not found: {p}")
    with open(p, "r") as f:
        data = json.load(f)
    # Ensure float cast where possible
    out: Dict[str, float] = {}
    for k, v in data.items():
        try:
            out[k] = float(v)
        except (ValueError, TypeError):
            pass
    return out
