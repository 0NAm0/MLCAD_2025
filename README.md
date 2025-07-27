# RL Framework for MLCAD25

**Status:** RL scaffold is implemented and runs with a mock backend. 

---

## 1. Purpose

Turn the EDA flow (OpenROAD / GLOAM / CircuitOps) into a reinforcement-learning (RL) environment:

1. Agent picks tool parameters (action)  
2. Tools run and produce QoR metrics (observation)  
3. Reward is computed from QoR  
4. Repeat for many steps/episodes

A mock runner lets us develop/test RL logic before the real pipeline is ready.

---

## 2. Directory Structure

```text
rl/
├─ agents/                               # RL algorithms
│  ├─ __init__.py
│  ├─ base_agent.py                      # Abstract agent interface
│  └─ dqn_agent.py                       # Minimal DQN (discrete actions)
│
├─ configs/
│  └─ config_example.yaml                # Example config (env/agent/trainer/reward)
│
├─ envs/
│  ├─ __init__.py
│  ├─ base_env.py                        # Abstract Env (reset/step)
│  └─ openroad_env.py                    # Env: idx action → params → tool → QoR → reward
│
├─ interfaces/
│  ├─ actions.yaml                       # Action schema: param names/types/ranges
│  ├─ qor_schema.yaml                    # QoR schema: metric names/units/better-worse
│  └─ README_interfaces.md               # Contract doc for RL ↔ HW
│
├─ memory/
│  ├─ __init__.py
│  └─ replay_buffer.py                   # FIFO buffer for off‑policy RL
│
├─ runners/
│  ├─ __init__.py
│  ├─ mock_runner.py                     # Fake QoR generator (no tool dependency)
│  └─ tool_runner.py                     # Real tool wrapper + QoR parsing hooks
│
├─ scripts/
│  ├─ __init__.py
│  ├─ run_openroad_with_params.py        # Called via `openroad -python`; reads params JSON, runs flow, dumps QoR
│  ├─ qor_parser.py                      # Parse QoR from stdout/JSON
│  ├─ train.py                           # Training entry
│  ├─ eval.py                            # Evaluation entry
│  └─ debug_spaces.py                    # Sanity check for action space/encoder
│
├─ trainers/
│  ├─ __init__.py
│  └─ trainer.py                         # Generic DQN-style training loop
│
└─ utils/
   ├─ __init__.py
   ├─ action_encoder.py                  # Discrete action index ↔ param dict (from actions.yaml)
   ├─ reward_fn.py                       # Reward calculator (weights/normalization)
   ├─ schema_loader.py                   # YAML schema loaders
   ├─ logger.py                          # Stdout + JSONL logger
   └─ config.py                          # YAML config loader
```



### Module Flow

1. **Agent** outputs a discrete action index.  
2. **OpenRoadEnv** decodes it via `ActionEncoder` → param dict → calls a **Runner**.  
3. **Runner**:
   - `MockRunner`: returns fake QoR, or  
   - `ToolRunner.run_openroad()`: writes params JSON → calls `run_openroad_with_params.py` using `openroad -python` → parses QoR with `qor_parser.py`.  
4. **Env** computes reward via `RewardFn` and returns `(obs, reward, done, info)`.  
5. **Trainer** manages experience collection + learning with `ReplayBuffer` and the `Agent`.

---

## 3. Current Capabilities

- Mock pipeline fully works (no dependency on HW).
- Discrete action space auto-generated from `actions.yaml`.
- Reward function configurable via YAML (no code change).
- Runs inside Apptainer (`/workspace` bind).
- Easy to swap/extend agents (PPO/SAC, etc.).

---

## 4. What We Need From the Hardware/Tool Team

Please confirm/adjust and provide:

1. **Action Space Specification**  
   - Final parameter list: names, types (float/int/categorical/bool), valid ranges/steps.  
   - Constraints/dependencies among parameters.  
   - Preferred passing method: JSON file, CLI flags, Tcl variables, etc.

2. **QoR Output Contract**  
   - Exact metrics we will receive (wirelength, slack, power, congestion, etc.).  
   - Output format (JSON, stdout `key=value`, or report file paths) + a sample output.

3. **Execution Flow / Scripts**  
   - The command(s) or scripts to run per RL step.  
   - Episode reset procedure (how to clean/import design each step).  
   - Typical runtime per run; parallelism allowed? timeouts/quotas?

4. **Paths & Environment**  
   - Final tool/script locations (OpenROAD/GLOAM/CircuitOps).  
   - Required environment variables/config files/licenses.

**After we get these:**

- Update `actions.yaml` / `qor_schema.yaml`.  
- Implement real param→Tcl/CLI mapping in `run_openroad_with_params.py`.  
- Parse real QoR reports in `qor_parser.py`.  
- Tune reward normalization/weights; start baseline RL runs.

---

## 5. How RL Uses Your Data

1. Agent selects parameters → saved to JSON.  

2. Command executed inside container:

   ```bash
   cd /workspace
   /opt/mlcad/MLCAD25-Contest-Scripts-Benchmarks/OpenROAD/build/src/openroad \
       -python rl/scripts/run_openroad_with_params.py \
       --params /workspace/tmp/params.json \
       --design ac97_top \
       --out_qor /workspace/tmp/qor.json

3. Tool flow runs, generates QoR (stdout/JSON/reports).

4. Parser extracts metrics, Env computes reward, returns to RL loop.

## 6. Run (Mock Mode)
# Inside Apptainer at /workspace
python rl/scripts/debug_spaces.py     # Inspect action space size
python rl/scripts/train.py            # Train with mock QoR

## 7. Possible Next Step
 Replace placeholders in actions.yaml and qor_schema.yaml with finalized specs.
 Implement real param→tool mapping & QoR parsing.
 Add unit tests (pytest) for encoder/parser/reward/runners.
 Provide rule-based/grid-search baselines for comparison.

