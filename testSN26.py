import json
import time

from sn26gnn import Reward

JSON_PATH = "example_trajectory.json"
CONTEXTS = [
    "A delivery robot is navigating in a museum.",
    "A museum robot is looking for a fire extinguisher, as it just detected a fire."
]

metric = Reward()

with open(JSON_PATH, "r") as input_file:
    trajectory_data = json.load(input_file)
 
for context in CONTEXTS:
    time_a = time.time()
    reward = metric.compute_reward(trajectory_data, context)
    time_b = time.time()
    print(f"The reward provided for {JSON_PATH} is {reward} assuming \"{context}\"")
    print(f"time: {time_b-time_a}")