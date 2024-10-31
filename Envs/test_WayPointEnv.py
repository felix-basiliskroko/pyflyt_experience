import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from tqdm import tqdm

import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

# id = "QuadX-Waypoint-v0"
id = "SingleWaypointQuadXEnv-v0"

env = gymnasium.make(id, render_mode=None)
# env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)

term, trunc = False, False
obs, _ = env.reset()
print(f'Current waypoint: {env.unwrapped.waypoints.targets[0]}')

while not (term or trunc):
    # new_action = env.action_space.sample()
    new_action = np.array([0.0, 0.0, 0.0, 0.0])
    # new_action = np.array([0.8, 0.8, 0.8, 0.8])
    obs, rew, term, trunc, _ = env.step(new_action)
    reward_info = env.reward_info
    print(f'Current altitude: {obs["altitude"]} ("scaled_altitude_reward": {reward_info["scaled_altitude_reward"]})')
    print(f'Action - aux_state: {new_action - obs["aux_state"]} - its L2 norm: {np.linalg.norm(new_action - obs["aux_state"])}; ("scaled_smooth_reward": {reward_info["scaled_smooth_reward"]})')
    print(f'Reward: {rew}')


env.reset()

