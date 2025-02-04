from stable_baselines3 import DDPG
from Envs import register
import numpy as np
import torch as t
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG

# Logdir
eval_freq = 30_000

# For Mac-Machine:
log_root_dir = "./logs/tensorboard_log/StaticWaypointEnv"
check_root_dir = "./checkpoints/StaticWaypointEnv"

# For Windows-Machine:
# log_root_dir = "../logs/tensorboard_log/StaticWaypointEnv"
# check_root_dir = "../checkpoints/StaticWaypointEnv"
run = "SingleWaypointNavigation"
mod = "DDPG_PosReward"
dir = f'{log_root_dir}/{run}/{mod}'

env_id = "SingleWaypointQuadXEnv-v0"
vec_env = make_vec_env(env_id=env_id, n_envs=10, seed=69)
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(4), sigma=float(0.5) * np.ones(4))

eval_env = make_vec_env(env_id=env_id, seed=42)
eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}",
                             log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                             deterministic=True, render=True, n_eval_episodes=5)
device = "cuda" if t.cuda.is_available() else "cpu"

model = DDPG("MultiInputPolicy", env=vec_env, tensorboard_log=dir, verbose=1, action_noise=action_noise, policy_kwargs=policy_kwargs)
model.learn(total_timesteps=2_000_000, callback=eval_callback)
