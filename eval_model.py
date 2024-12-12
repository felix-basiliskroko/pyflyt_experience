from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np

env_id = "SingleWaypointQuadXEnv-v0"
env = gym.make(env_id, render_mode=None)
agent_pos = []

term, trunc = False, False
obs, info = env.reset()
ep_reward = 0


for _ in range(1):
    # Evaluate the agent
    while not (term or trunc):
        action = env.action_space.sample()
        obs, rew, term, trunc, info = env.step(action)
        print(f'Weighted LOS reward: {info["reward_components"]["w_los_reward"]}')
        print(f'Weighted smooth reward: {info["reward_components"]["w_los_smooth_reward"]}')
        print(rew)
    env.reset()
