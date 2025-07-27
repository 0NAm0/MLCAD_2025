#!/usr/bin/env python3
"""
Run inside the Apptainer container.

Usage (from /workspace):
  /opt/mlcad/.../openroad -python rl/scripts/run_openroad_with_params.py \
      --params /workspace/tmp/params.json \
      --design ac97_top

What it does:
1. Read a JSON of action parameters.
2. (TODO) Translate them to OpenROAD commands / Tcl / flags.
3. Invoke the needed flows (place/route/etc.).
4. Collect QoR numbers and print them as "KEY=VALUE" lines or JSON.

Right now it's a stub: it just prints fake QoR using the params.
Replace the TODO sections when HW flow is ready.
"""

import argparse
import json
import sys
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--params", type=str, required=True,
                   help="Path to JSON file containing action parameters.")
    p.add_argument("--design", type=str, required=True,
                   help="Design name (e.g., ac97_top).")
    p.add_argument("--out_qor", type=str, default="/workspace/tmp/qor.json",
                   help="Where to dump QoR JSON (optional).")
    return p.parse_args()

def main():
    args = parse_args()
    params_path = Path(args.params)
    if not params_path.exists():
        print(f"[ERROR] Param file not found: {params_path}", file=sys.stderr)
        sys.exit(1)

    with open(params_path, "r") as f:
        action_params = json.load(f)

    # TODO: Generate Tcl / config files or CLI flags for OpenROAD here.
    # Example (pseudo):
    #   tcl_script = generate_tcl(action_params, design=args.design)
    #   run_openroad_flow(tcl_script)

    # TODO: After the flow finishes, parse real QoR from reports.
    # Placeholder QoR (replace with real parsing):
    qor = {
        "wirelength": 1.23e6,
        "timing_slack": -0.05,
        "power_dynamic": 150.2,
        "power_leakage": 12.3,
        "congestion_max": 0.12
    }

    # Print key=value so stdout parser works
    for k, v in qor.items():
        print(f"{k}={v}")

    # Optional: also store as JSON file
    with open(args.out_qor, "w") as fo:
        json.dump(qor, fo)

if __name__ == "__main__":
    main()
