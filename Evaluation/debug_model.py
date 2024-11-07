import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from stable_baselines3 import PPO
from tqdm import tqdm

from Envs.register import register  # needed to register the custom environments in gymnasium

# id = "QuadX-Waypoint-v0"
id = "SingleWaypointQuadXEnv-v0"
deterministic = False
moel_path = "./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted/best_model"

env = gymnasium.make(id, render_mode=None)
model = PPO("MultiInputPolicy", env=env)
model.load(moel_path, deterministic=deterministic)
# env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)

term, trunc = False, False
obs = env.reset()[0]
states = None
prev_aux_state = obs["aux_state"]

for _ in tqdm(range(1)):
    while not (term or trunc):
        actions, states = model.predict(obs, deterministic=deterministic, state=states)
        new_obs, rew, done, info = env.step(actions)
        print(new_obs)
        reward_info = env.unwrapped.reward_info

        print(f'Action - aux_state: {actions - prev_aux_state} - its L2 norm: {np.linalg.norm(actions - prev_aux_state)}; ("scaled_smooth_reward": {reward_info["scaled_smooth_reward"]})')
        print(f'Absolute difference in azimuth angle: {np.abs(obs["t_azimuth_angle"] - obs["a_azimuth_angle"])}')
        print(f'Absolute difference in elevation angle: {np.abs(obs["t_elevation_angle"] - obs["a_elevation_angle"])}')
        print(f'los_reward: {reward_info["scaled_los_reward"]}')
        print("-------------------------------------------------------------------------------------------------------")
        print(f'Reward: {rew}')
        obs = new_obs
        prev_aux_state = obs["aux_state"]
    env.reset()
    prev_aux_state = obs["aux_state"]
