import datetime
import torch as t
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
from stable_baselines3.common.vec_env import VecVideoRecorder

import Envs.register
from scheduler.scheduling import linear_schedule, exponential_schedule, cosine_annealing_schedule

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import StabilityEvalCallback
from stable_baselines3.common.callbacks import EvalCallback

# Logdir
eval_freq = 30_000

# For Mac-Machine:
log_root_dir = "./logs/final_log/StaticWaypointEnv"
check_root_dir = "./checkpoints/StaticWaypointEnv"

# For Windows-Machine:
# log_root_dir = "../logs/final_log/StaticWaypointEnv"
# check_root_dir = "../checkpoints/StaticWaypointEnv"
num_runs = 1

run = "SingleWaypointNavigation"
mod = "FullyTunedOnlyLOS"
dir = f'{log_root_dir}/{run}/{mod}'

for i in range(num_runs):
    env_id = "SingleWaypointQuadXEnv-v0"
    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

    eval_env = make_vec_env(env_id=env_id, seed=42)
    eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}/best_model_run_{i}",
                     log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                     deterministic=True, render=True, n_eval_episodes=5)
    device = "cuda" if t.cuda.is_available() else "cpu"

    # lr = 7e-3
    lr = 8e-4
    # lr = linear_schedule(initial_lr=lr)
    # lr = exponential_schedule(initial_lr=lr, decay_rate=0.99)
# lr = cosine_annealing_schedule(initial_lr=8e-3, min_lr=1e-3)

    '''
    model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
                ent_coef=0.007,
                gamma=0.863,
                learning_rate=8e-4,
                gae_lambda=0.9625,
                device=device)  # For non-dict observation space
    model.learn(total_timesteps=250_000, callback=eval_callback)
    '''
    model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
                batch_size=1024,
                ent_coef=0.0,
                gamma=0.985,
                learning_rate=lr,
                gae_lambda=0.9875,
                device=device)  # For non-dict observation space
    model.learn(total_timesteps=250_000, callback=eval_callback)
