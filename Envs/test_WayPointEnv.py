import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from tqdm import tqdm

import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

# id = "QuadX-Waypoint-v0"
id = "SingleWaypointQuadXEnv-v0"

env = gymnasium.make(id, render_mode="human")
# env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)

term, trunc = False, False
obs, _ = env.reset()
print(f'Current waypoint: {env.unwrapped.waypoints.targets[0]}')

while not (term or trunc):
    new_action = env.action_space.sample()
    # new_action = np.array([0.8, 0.8, 0.8, 0.8])
    obs, rew, term, trunc, _ = env.step(new_action)

env.reset()

