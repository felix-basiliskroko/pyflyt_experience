import torch as t
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
import Envs.register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Logdir
time_steps = 85_000
vec_env = make_vec_env("Quadx-Waypoint-v0", n_envs=1)
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

for gamma_coeff in [0.8, 0.9, 0.95, 0,90]:
    for vf_coeff in [0.5, 0.7, 0.9]:
        for ent_coeff in [0.001, 0.005, 0.01]:
            for lr in [1e-7, 1e-5, 1e-3]:
                dir = f'./hyperparam_log/gamma_{gamma_coeff}_vf_{vf_coeff}_ent_{ent_coeff}_lr_{lr}'
                model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
                            ent_coef=ent_coeff,
                            vf_coef=vf_coeff,
                            gamma=gamma_coeff,
                            learning_rate=lr)
                model.learn(total_timesteps=time_steps)