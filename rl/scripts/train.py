import sys
import os

# Allow "python rl/scripts/train.py" to import rl.*
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.utils.config import load_cfg
from rl.envs.openroad_env import OpenRoadEnv
from rl.agents.dqn_agent import DQNAgent
from rl.trainers.trainer import Trainer
from rl.utils.logger import Logger

def main(cfg_path: str = "rl/configs/config_example.yaml"):
    cfg = load_cfg(cfg_path)

    # 1. Build env
    env_cfg = cfg["env"]
    env = OpenRoadEnv(**env_cfg)

    # 2. Build agent (placeholder dims)
    agent_cfg = cfg["agent"]
    agent = DQNAgent(agent_cfg, obs_dim=3, act_dim=11)

    # 3. Trainer
    trainer_cfg = cfg["trainer"]
    logger = Logger(out_dir="./rl_logs")
    trainer = Trainer(env, agent, trainer_cfg, logger)

    # 4. Train (stub; won't run properly until HW pipeline ready)
    trainer.train()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
