import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from tqdm import tqdm

import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

# env = gymnasium.make("Quadx-Waypoint-v0", render_mode="human")
env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)

term, trunc = False, False
obs, _ = env.reset()
print(f'Current waypoint: {env.waypoint}')

while not (term or trunc):
    # new_action = env.action_space.sample()
    new_action = np.array([0.0, 0.0, 0.0, 0.0])
    obs, rew, term, trunc, _ = env.step(new_action)
    # print(f"Distance to waypoint: {np.linalg.norm(obs['target_delta'])}; Reward: {rew}")
env.reset()

