import datetime
import torch as t
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
from stable_baselines3.common.vec_env import VecVideoRecorder

import Envs.register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import StabilityEvalCallback
from stable_baselines3.common.callbacks import EvalCallback

# Logdir
eval_freq = 30_000
log_root_dir = "./logs/tensorboard_log/StaticWaypointEnv"
check_root_dir = "./checkpoints/StaticWaypointEnv"
run = "SingleWaypointNavigation"
mod = "AngPosMovement"
dir = f'{log_root_dir}/{run}/{mod}'

env_id = "SingleWaypointQuadXEnv-v0"
vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

eval_env = make_vec_env(env_id=env_id, seed=42)
eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}",
                 log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                 deterministic=True, render=True, n_eval_episodes=5)
device = "cuda" if t.cuda.is_available() else "cpu"


model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
            ent_coef=0.007,
            gamma=0.863,
            learning_rate=8e-4,
            gae_lambda=0.9625,
            device=device)  # For non-dict observation space
model.learn(total_timesteps=2_000_000, callback=eval_callback)
