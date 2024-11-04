from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np

env_id = "SingleWaypointQuadXEnv-v0"
env = gym.make(env_id, render_mode="human")
model = PPO("MultiInputPolicy", env=env)
model.load("./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted-OrnHeight/best_model", deterministic=True)
agent_pos = []

term, trunc = False, False
obs, info = env.reset()
ep_reward = 0


for _ in range(1):
    # Evaluate the agent
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=True)
        # action = action.squeeze(0)
        obs, rew, term, trunc, _ = env.step(action)
        reward_info = env.reward_info
        print(f'Current altitude: {obs["altitude"]} ("scaled_altitude_reward": {reward_info["scaled_altitude_reward"]})')
        print(f'Action - aux_state: {action - obs["aux_state"]} - its L2 norm: {np.linalg.norm(action - obs["aux_state"])}; ("scaled_smooth_reward": {reward_info["scaled_smooth_reward"]})')
        print(f'Difference in azimuth angle: {np.abs(obs["t_azimuth_angle"] - obs["a_azimuth_angle"])}')
        print(f'Difference in elevation angle: {np.abs(obs["t_elevation_angle"] - obs["a_elevation_angle"])}')
        print(f'los_reward: {reward_info["scaled_los_reward"]}')
        print(f'Total reward: {rew}')
        print("-------------------------------------------------------------------------------------------------------")

        ep_reward += rew

    print(f'Episode reward: {ep_reward}')
    env.reset()
