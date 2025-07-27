from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Sequence
import itertools
import numpy as np

@dataclass
class ParamChoice:
    name: str
    values: List[Any]        # enumerated discrete values

class ActionEncoder:
    """
    Convert between:
      - discrete action index (0..N-1)
      - action dict {param: value}
    by enumerating each param's discrete choices (floats quantized by 'step').
    """

    def __init__(self, action_schema: Dict[str, Any]):
        self.params: List[ParamChoice] = []
        for item in action_schema["actions"]:
            name = item["name"]
            typ = item["type"]
            if typ in ("float", "int"):
                step = item.get("step")
                if step is None:
                    raise ValueError(f"Param {name} requires 'step' for discretization.")
                lo = item["min"]; hi = item["max"]
                if typ == "int":
                    vals = list(range(int(lo), int(hi) + 1, int(step)))
                else:
                    count = int(round((hi - lo) / step)) + 1
                    vals = [round(lo + i * step, 10) for i in range(count)]
            elif typ == "categorical":
                vals = item["values"]
            elif typ == "bool":
                vals = [False, True]
            else:
                raise ValueError(f"Unsupported type: {typ}")
            self.params.append(ParamChoice(name=name, values=vals))

        # Precompute Cartesian product sizes for index mapping
        self.radices = [len(p.values) for p in self.params]
        self.num_actions = int(np.prod(self.radices)) if self.radices else 0

    def index_to_dict(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_actions:
            raise IndexError(f"Action index {idx} out of range 0..{self.num_actions-1}")
        result = {}
        for p in reversed(self.params):
            base = len(p.values)
            choice = idx % base
            result[p.name] = p.values[choice]
            idx //= base
        return result

    def dict_to_index(self, action_dict: Dict[str, Any]) -> int:
        idx = 0
        mult = 1
        for p in self.params:
            val = action_dict[p.name]
            try:
                c = p.values.index(val)
            except ValueError:
                raise ValueError(f"Value {val} not in choices of param {p.name}")
            idx += c * mult
            mult *= len(p.values)
        return idx

    def sample_random_index(self) -> int:
        return np.random.randint(0, self.num_actions)

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": p.name, "choices": p.values} for p in self.params]
