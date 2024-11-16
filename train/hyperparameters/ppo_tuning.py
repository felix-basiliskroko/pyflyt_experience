import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch as t
from Envs import register


def entr_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, env_id, "entropy")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_dir = f"{log_dir}/v_range={value_range}_buckets={buckets}_steps={num_steps}"
    ent_coeff_vals = np.linspace(value_range[0], value_range[1], buckets)

    for ent_coeff in ent_coeff_vals:
        run_dir = f"{log_dir}/ent_coeff={round(ent_coeff, 3)}"
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=run_dir,
                    policy_kwargs=policy_kwargs,
                    ent_coef=ent_coeff,
                    device=device)
        model.learn(total_timesteps=num_steps)
    print("Finished tuning entropy coefficient.")


def gamma_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, env_id, "gamma")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_dir = f"{log_dir}/v_range={value_range}_buckets={buckets}_steps={num_steps}"
    gamma_vals = np.linspace(value_range[0], value_range[1], buckets)

    for gamma in gamma_vals:
        run_dir = f"{log_dir}/gamma={round(gamma, 3)}"
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=run_dir,
                    policy_kwargs=policy_kwargs,
                    ent_coef=0.007,
                    gamma=gamma,
                    device=device)
        model.learn(total_timesteps=num_steps)
    print("Finished tuning gamma.")


def gae_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, env_id, "gamma")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_dir = f"{log_dir}/v_range={value_range}_buckets={buckets}_steps={num_steps}"
    gae_vals = np.linspace(value_range[0], value_range[1], buckets)

    for gae in gae_vals:
        run_dir = f"{log_dir}/gamma={round(gae, 3)}"
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=run_dir,
                    policy_kwargs=policy_kwargs,
                    ent_coef=0.007,
                    gamma=0.863,
                    gae_lambda=gae,
                    device=device)
        model.learn(total_timesteps=num_steps)
    print("Finished tuning gae_lambda.")