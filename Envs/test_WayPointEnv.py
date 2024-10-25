import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from tqdm import tqdm

import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)
# env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)
ang_pos_min, ang_pos_max = np.inf, -np.inf

term, trunc = False, False
obs, _ = env.reset()
print(f'Current waypoint: {env.waypoint}')

while not (term or trunc):
    new_action = env.action_space.sample()
    # new_action = np.array([0.8, 0.8, 0.8, 0.8])
    obs, rew, term, trunc, _ = env.step(new_action)
    info_state = env.unwrapped.get_info_state()
    if info_state["ang_pos"].any() < ang_pos_min:
        ang_pos_min = np.min(info_state["ang_pos"])
    if info_state["ang_pos"].any() > ang_pos_max:
        ang_pos_max = np.max(info_state["ang_pos"])
    # print(f'Current reward: {rew}')
env.reset()
print(f'Mini: {ang_pos_min}; Max: {ang_pos_max}')

