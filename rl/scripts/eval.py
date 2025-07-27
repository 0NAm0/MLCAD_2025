import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rl.utils.config import load_cfg
from rl.envs.openroad_env import OpenRoadEnv
from rl.agents.dqn_agent import DQNAgent

def main(cfg_path: str = "rl/configs/config_example.yaml", model_path: str = "./rl_logs/dqn.pt"):
    cfg = load_cfg(cfg_path)
    env = OpenRoadEnv(**cfg["env"])

    agent = DQNAgent(cfg["agent"], obs_dim=3, act_dim=11)
    agent.load(model_path)

    obs = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action = agent.select_action(obs)   # here epsilon still decays; set to greedy if needed
        obs, reward, done, info = env.step(action)
        total_reward += reward

    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
