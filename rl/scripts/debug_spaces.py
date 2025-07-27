import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path: sys.path.insert(0, ROOT)

from rl.utils.schema_loader import load_actions_schema
from rl.utils.action_encoder import ActionEncoder

if __name__ == "__main__":
    schema = load_actions_schema("rl/interfaces/actions.yaml")
    enc = ActionEncoder(schema)
    print("num_actions =", enc.num_actions)
    print("first 3 actions:")
    for i in range(min(3, enc.num_actions)):
        print(i, enc.index_to_dict(i))
