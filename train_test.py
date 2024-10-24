import datetime
import torch as t
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
import Envs.register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import StabilityEvalCallback
from stable_baselines3.common.callbacks import EvalCallback

# Logdir
eval_freq = 20_000
log_root_dir = "./tensorboard_log/StaticWaypointEnv"
run = "SimpleObs"
mod = "TargetDelta-Radius-1000m"
dir = f'{log_root_dir}/{run}/{mod}'
vec_env = make_vec_env("Quadx-Waypoint-v0", n_envs=1)
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

eval_env = gym.make("Quadx-Waypoint-v0", render_mode=None)
eval_callback = EvalCallback(eval_env, best_model_save_path=f"./checkpoints/{run}/{mod}",
                 log_path=f"./checkpoints/{run}/{mod}", eval_freq=eval_freq,
                 deterministic=True, render=False)

model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
            ent_coef=0.005,
            gamma=0.95,
            learning_rate=3e-6)  # For non-dict observation space
model.learn(total_timesteps=500_000, callback=eval_callback)
