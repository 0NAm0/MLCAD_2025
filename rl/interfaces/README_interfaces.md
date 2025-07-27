# Interface Contracts for RL ⇄ EDA Tools

This document defines the *contract* between the RL side and the hardware/tool side.

## 1. Actions (Input to Tools)

- File: `actions.yaml`
- Each entry defines:
  - `name`: parameter key
  - `type`: float/int/categorical/bool
  - Range or enum values
  - `default`
  - Optional `step` for discretization
- RL agent will output a dict `{name: value}` per step.
- HW scripts must accept these values (e.g., via CLI args, env vars, or a JSON file).

## 2. QoR Metrics (Output from Tools)

- File: `qor_schema.yaml`
- HW side must return these metrics for every run.
- Format suggested: **JSON** file or printed key=value lines to stdout.
- RL parser will map those keys to a dict and compute rewards.

## 3. Call Flow (High-level)

1. RL agent decides `action_dict`.
2. RL writes `action_dict` to a temp JSON file or passes via CLI.
3. A runner script:
   - Calls OpenROAD/GLOAM/etc. with those params.
   - Collects QoR metrics from tool outputs.
4. Runner returns a dict to the env, reward is computed.

## 4. To Be Clarified with HW Team

- Confirm exact parameter names and their legal ranges.
- Confirm tool execution command and where outputs (QoR) can be found.
- Confirm average runtime per run and whether parallel runs are allowed.
- Confirm reset/initialization procedure for each episode.

## 5. Versioning

- Update `metadata.version` when schemas change.
- Keep backward compatibility or write a migration note.

