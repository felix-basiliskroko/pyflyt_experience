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
from Evaluation.custom_eval_callback import CustomEvalCallback

# Logdir
num_runs = 5
total_steps = 400_000
eval_freq = 20_000

log_root_dir = "./logs/tensorboard_log/StaticWaypointEnv"
check_root_dir = "./checkpoints/StaticWaypointEnv"
run = "SingleWaypointNavigation"
mod = "DebugDefaultHyperModAltitude"
dir = f'{log_root_dir}/{run}/{mod}'

env_id = "SingleWaypointQuadXEnv-v0"
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
shift = 0.75
# shifts = [0.0, 0.25, 0.5, 0.75, 1.0]

# for s in shifts:
for r in range(num_runs):
    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": shift, "steep_grad": 1.0})
    eval_env = gym.make(env_id, reward_shift=shift, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}",
                     log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                     deterministic=True, render=True, n_eval_episodes=100)
    device = "cuda" if t.cuda.is_available() else "cpu"

    model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=f'{dir}', policy_kwargs=policy_kwargs,
                device=device)  # For non-dict observation space
    model.learn(total_timesteps=total_steps, callback=eval_callback)
