import json
import os
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import torch as t
import time

from Envs import register
from Evaluation.custom_eval_callback import CustomEvalCallback
from Evaluation.vis_model import aggregate_eval
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def batch_tune(env_id: str, log_dir: str, values: list, num_steps=300_000):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": 0.75, "steep_grad": 1.0})
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
    log_dir = os.path.join(log_dir, "batch_size")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for b_size in values:
        model = DDPG("MultiInputPolicy",
                     env=vec_env,
                     verbose=0,
                     tensorboard_log=log_dir,
                     batch_size=b_size,
                     buffer_size=1_000_000,
                     policy_kwargs=policy_kwargs,
                     device=device)
        model.learn(total_timesteps=num_steps, tb_log_name=f'batch_size={b_size}', callback=eval_callback)
    print("Finished tuning batch size.")


def action_noise_tune(env_id: str, log_dir: str, values: list, num_steps=300_000):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": 0.75, "steep_grad": 1.0})
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
    log_dir = os.path.join(log_dir, "action_noise")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for noise in values:
        if isinstance(noise, NormalActionNoise):
            tb_log_name = f'action_noise=NormalActionNoise'
        elif isinstance(noise, OrnsteinUhlenbeckActionNoise):
            tb_log_name = f'action_noise=OrnsteinUhlenbeckActionNoise'
        elif noise is None:
            tb_log_name = f'action_noise=NoNoise'
        else:
            raise ValueError(f"Invalid action noise type: {noise}")

        model = DDPG("MultiInputPolicy",
                     env=vec_env,
                     verbose=0,
                     tensorboard_log=log_dir,
                     batch_size=128,
                     buffer_size=1_000_000,
                     action_noise=noise,
                     policy_kwargs=policy_kwargs,
                     device=device)
        model.learn(total_timesteps=num_steps, tb_log_name=tb_log_name, callback=eval_callback)
    print("Finished tuning action noise type.")


def lr_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps=300_000):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": 0.75, "steep_grad": 1.0})
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
    log_dir = os.path.join(log_dir, "lr")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    lr_vals = np.linspace(value_range[0], value_range[1], buckets)

    for rate in lr_vals:
        model = DDPG("MultiInputPolicy",
                     env=vec_env,
                     verbose=0,
                     tensorboard_log=log_dir,
                     batch_size=128,
                     buffer_size=1_000_000,
                     action_noise=None,
                     learning_starts=100,
                     learning_rate=rate,
                     policy_kwargs=policy_kwargs,
                     device=device)
        model.learn(total_timesteps=num_steps, tb_log_name=f'lr={round(rate, 5)}', callback=eval_callback)
    print("Finished tuning the entropy coefficient.")
