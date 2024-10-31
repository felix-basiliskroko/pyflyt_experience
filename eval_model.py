from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np


env = gym.make("Quadx-Waypoint-v0", render_mode=None)
model = PPO("MultiInputPolicy", env=env)
model.load("./checkpoints/SimpleObs/Smooth-Control-Reward-v2/best_model", deterministic=False)
agent_pos = []

term, trunc = False, False
obs, info = env.reset()
ep_reward = 0


for _ in range(1):
    # Evaluate the agent
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=False)
        action = action.squeeze(0)
        obs, rew, term, trunc, _ = env.step(action)
        info_state = env.get_info_state()
        rewards = info_state['reward']
        print(f'Distance to target: {np.linalg.norm(info_state["lin_pos"] - env.waypoint)} (Distance_target_reward: {rewards["distance_target_reward"]})')
        print(f'altitude: {info_state["lin_pos"][2]} (distance_ground_reward: {rewards["distance_ground_reward"]})')
        print(f'Prev action - action: {obs["auxiliary"] - action}, (smooth_control_reward: {rewards["smooth_control_reward"]})')
        print(f'Angular velocity: {obs["angular_vel"]}, (smooth_ang_vel_reward: {rewards["smooth_ang_vel_reward"]})')
        print("-------------------------------------------------------------------------------------------------------")

        agent_pos.append(info_state['lin_pos'])

        ep_reward += rew

    print(f'Episode reward: {ep_reward}')
    env.reset()
