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
from Evaluation.custom_eval_callback import CustomEvalCallback


def run_ppo_training(num_runs, total_steps, eval_freq, shift, env_id, policy_kwargs, run, mod, dir, check_root_dir, lr_mode, lr, hyperparam_mode):
    if lr_mode == "linear":
        lr = linear_schedule(initial_lr=lr)
    elif lr_mode == "exponential":
        lr = exponential_schedule(initial_lr=lr, decay_rate=0.99)
    elif lr_mode == "cosine":
        lr = cosine_annealing_schedule(initial_lr=lr, min_lr=5e-6)
    elif lr_mode == "constant":
        lr = lr
    else:
        raise ValueError(f"Invalid learning rate mode: {lr_mode}")

    for r in range(num_runs):
        vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": shift, "steep_grad": 1.0})
        eval_env = gym.make(env_id, reward_shift=shift, steep_grad=1.0)
        eval_callback = CustomEvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}/run_{r}",
                         log_path=f"./{check_root_dir}/{run}/{mod}/run{r}", eval_freq=eval_freq,
                         deterministic=True, render=True, n_eval_episodes=100)
        device = "cuda" if t.cuda.is_available() else "cpu"

        if hyperparam_mode == "default":
            model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=f'{dir}', policy_kwargs=policy_kwargs,
                        device=device)
        elif hyperparam_mode == "tuned":
            model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=f'{dir}', policy_kwargs=policy_kwargs,
                        batch_size=256,
                        ent_coef=0.009,
                        gamma=0.975,
                        gae_lambda=0.94,
                        learning_rate=lr,
                        device=device)
        else:
            raise ValueError(f"Invalid hyperparameter mode: {hyperparam_mode}")
        model.learn(total_timesteps=total_steps, callback=eval_callback, tb_log_name=f'hyperparam_{hyperparam_mode}_lr_mode_{lr_mode}_run')
