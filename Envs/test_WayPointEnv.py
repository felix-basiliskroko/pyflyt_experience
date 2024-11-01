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
prev_aux_state = obs["aux_state"]
print(f'Previous aux_state: {prev_aux_state}')
print(f'Current waypoint: {env.unwrapped.waypoints.targets[0]}')
smooth_rew_min, smooth_rew_max = np.inf, -np.inf
los_rew_min, los_rew_max = np.inf, -np.inf


for _ in tqdm(range(10_000)):
    while not (term or trunc):
        new_action = env.action_space.sample()
        # new_action = np.array([0.0, 0.0, 0.0, 0.0])
        # new_action = np.array([np.pi, np.pi, np.pi, 0.8])
        # new_action = np.array([0.8, 0.8, 0.8, 0.8])
        obs, rew, term, trunc, _ = env.step(new_action)
        reward_info = env.reward_info
        if reward_info["scaled_smooth_reward"] < smooth_rew_min:
            smooth_rew_min = reward_info["scaled_smooth_reward"]
        if reward_info["scaled_smooth_reward"] > smooth_rew_max:
            smooth_rew_max = reward_info["scaled_smooth_reward"]

        if reward_info["scaled_los_reward"] < los_rew_min:
            los_rew_min = reward_info["scaled_los_reward"]

        if reward_info["scaled_los_reward"] > los_rew_max:
            los_rew_max = reward_info["scaled_los_reward"]
        print(f'Action - aux_state: {new_action - prev_aux_state} - its L2 norm: {np.linalg.norm(new_action - prev_aux_state)}; ("scaled_smooth_reward": {reward_info["scaled_smooth_reward"]})')
        print(f'Absolute difference in azimuth angle: {np.abs(obs["t_azimuth_angle"] - obs["a_azimuth_angle"])}')
        print(f'Absolute difference in elevation angle: {np.abs(obs["t_elevation_angle"] - obs["a_elevation_angle"])}')
        print(f'los_reward: {reward_info["scaled_los_reward"]}')
        print("-------------------------------------------------------------------------------------------------------")
        print(f'Reward: {rew}')
        prev_aux_state = obs["aux_state"]
    env.reset()
    prev_aux_state = obs["aux_state"]

print(f'Minimum smooth reward: {smooth_rew_min}')
print(f'Maximum smooth reward: {smooth_rew_max}')

print(f'Minimum los reward: {los_rew_min}')
print(f'Maximum los reward: {los_rew_max}')

