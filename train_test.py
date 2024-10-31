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
check_root_dir = "./checkpoints/StaticWaypointEnv"
run = "SingleWaypointNavigation"
mod = "DistancePlusAltitudeReward"
dir = f'{log_root_dir}/{run}/{mod}'

# env_id = "PyFlyt/QuadX-Waypoints-v2"
env_id = "SingleWaypointQuadXEnv-v0"
# env_id = "Quadx-Waypoint-v0"


vec_env = make_vec_env(env_id=env_id, n_envs=1)
# vec_env = gym.make(env_id, render_mode=None)
vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

eval_env = make_vec_env(env_id=env_id)
eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=True, training=False)
eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}",
                 log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                 deterministic=True, render=False)
device = "cuda" if t.cuda.is_available() else "cpu"


model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
            ent_coef=0.0055,
            vf_coef=0.6,
            gamma=0.8,
            learning_rate=0.001,
            device=device)  # For non-dict observation space
model.learn(total_timesteps=2_000_000, callback=eval_callback)
