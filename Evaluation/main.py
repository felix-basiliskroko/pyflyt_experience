import gymnasium as gym
from stable_baselines3 import PPO

from Evaluation.vis_model import visualize_model

model_path = "../checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted-MAYBEFIXED/best_model"
env_id = "SingleWaypointQuadXEnv-v0"

render = True
num_eval_eps = 5

# Create model and environment
env = gym.make(env_id, render_mode="human" if render else None)
_ = env.reset()
model = PPO.load(model_path, deterministic=True)

visualize_model(model, env, num_eval_eps, render, verbose=False)