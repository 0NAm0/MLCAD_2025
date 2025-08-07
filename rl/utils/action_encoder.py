"""

After call `build_gate_op_table(instance_names)` once per episode,
`num_actions` becomes:

    (#param-actions)  +  (#gate-ops × #instances)

and each discrete index can be mapped to either a param-dict or a gate-op
dict of the form
    { "type": "gate_op",
      "op": "clone_buffer",
      "target_master": "BUF",
      "inst": "U1234" }
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


@dataclass
class ParamChoice:
    name: str
    values: List[Any]


class ActionEncoder:

    # constructor – parse actions.yaml (already loaded as dict)
    def __init__(self, action_schema: Dict[str, Any]):
        # split into param-type and gate-op specs
        self.param_specs: List[Dict[str, Any]] = []
        self.gate_op_specs: List[Dict[str, Any]] = []

        for item in action_schema["actions"]:
            if item["type"] == "gate_op":
                self.gate_op_specs.append(item)
            else:
                self.param_specs.append(item)

        self.params: List[ParamChoice] = []
        for item in self.param_specs:
            name = item["name"]
            typ = item["type"]

            if typ in ("float", "int"):
                step = item.get("step")
                if step is None:
                    raise ValueError(f"Param {name} requires 'step' for discretisation")
                lo, hi = item["min"], item["max"]
                if typ == "int":
                    vals = list(range(int(lo), int(hi) + 1, int(step)))
                else:  # float
                    n = int(round((hi - lo) / step)) + 1
                    vals = [round(lo + i * step, 10) for i in range(n)]
            elif typ == "categorical":
                vals = item["values"]
            elif typ == "bool":
                vals = [False, True]
            else:
                raise ValueError(f"Unsupported type: {typ}")

            self.params.append(ParamChoice(name=name, values=vals))

        self.radices = [len(p.values) for p in self.params]
        self._param_action_count = int(np.prod(self.radices)) if self.radices else 0

        # gate-op lookup table will be built at reset-time
        self.gate_op_table: List[tuple[Dict[str, Any], str]] = []
        self.num_actions = self._param_action_count  # provisional


    #  call once per episode after you know instance list
    def build_gate_op_table(self, instance_names: List[str]):
        self.gate_op_table.clear()
        for op_spec in self.gate_op_specs:
            for inst in instance_names:
                self.gate_op_table.append((op_spec, inst))
        self.num_actions = self._param_action_count + len(self.gate_op_table)

    # index  →  action-dict
    def index_to_dict(self, idx: int) -> Dict[str, Any]:
        if idx < 0 or idx >= self.num_actions:
            raise IndexError(f"index {idx} out of 0..{self.num_actions-1}")

        # parameter actions
        if idx < self._param_action_count:
            result = {}
            for p in reversed(self.params):
                base = len(p.values)
                choice = idx % base
                result[p.name] = p.values[choice]
                idx //= base
            result["type"] = "param"
            return result

        # gate-op actions
        op_spec, inst = self.gate_op_table[idx - self._param_action_count]
        return {
            "type": "gate_op",
            "op": op_spec["op"],
            "target_master": op_spec.get("target_master"),
            "result_masters": op_spec.get("result_masters"),
            "inst": inst,
        }

    # action-dict  →  index   (only needed if log or test)
    def dict_to_index(self, action_dict: Dict[str, Any]) -> int:
        if action_dict["type"] == "param":
            idx = 0
            mult = 1
            for p in self.params:
                c = p.values.index(action_dict[p.name])
                idx += c * mult
                mult *= len(p.values)
            return idx

        # gate-op
        key = (next(spec for spec in self.gate_op_specs if spec["op"] == action_dict["op"]),  # spec obj
               action_dict["inst"])
        off = self.gate_op_table.index(key)
        return self._param_action_count + off


    def sample_random_index(self) -> int:
        return np.random.randint(0, self.num_actions)

    def describe(self) -> List[Dict[str, Any]]:
        """Return human-readable list of parameters (not gate-ops)."""
        return [{"name": p.name, "choices": p.values} for p in self.params]
