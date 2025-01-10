import sys
import torch as t

from train_ppo import run_ppo_training
from train_sac import run_sac_training
from train_ddpg import run_ddpg_training

# Logdir
num_runs = 3
total_steps = 800_000
eval_freq = 20_000
shift = 0.75
# shifts = [0.0, 0.25, 0.5, 0.75, 1.0]

# For Mac-Machine:
log_root_dir = "../logs/tensorboard_log/Final/DDPG"
check_root_dir = "../checkpoints/Final/DDPG"

# For Windows-Machine:
# log_root_dir = "../logs/tensorboard_log/StaticWaypointEnv"
# check_root_dir = "../checkpoints/StaticWaypointEnv/final"

run = "SingleWaypointNavigation"
mod = "DefaultHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

env_id = "SingleWaypointQuadXEnv-v0"
# policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))  # For PPO
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))  # For SAC/DDPG
lr = 7e-5  # DDPG
# lr = 3e-4  # PPO
# lr = 3e-4  # SAC


run_ddpg_training(num_runs=num_runs,
                  total_steps=total_steps,
                  eval_freq=eval_freq,
                  shift=shift,
                  env_id=env_id,
                  policy_kwargs=policy_kwargs,
                  run=run,
                  mod=mod,
                  dir=dir,
                  check_root_dir=check_root_dir,
                  lr_mode="constant",
                  lr=lr,
                  hyperparam_mode="default",
                  flight_mode=-1)

mod = "MThrustTunedHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ddpg_training(num_runs=num_runs,
                  total_steps=total_steps,
                  eval_freq=eval_freq,
                  shift=shift,
                  env_id=env_id,
                  policy_kwargs=policy_kwargs,
                  run=run,
                  mod=mod,
                  dir=dir,
                  check_root_dir=check_root_dir,
                  lr_mode="linear",
                  lr=lr,
                  hyperparam_mode="tuned",
                  flight_mode=-1)

total_steps = 800_000  # For PPO
lr = 3e-4  # PPO
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))  # For PPO
log_root_dir = "../logs/tensorboard_log/Final/PPO"
check_root_dir = "../checkpoints/Final/PPO"
mod = "MThrustTunedHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="tuned",
                 flight_mode=-1)

total_steps = 1_000_000  # For SAC
lr = 3e-4  # SAC
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))  # For SAC
log_root_dir = "../logs/tensorboard_log/Final/SAC"
check_root_dir = "../checkpoints/Final/SAC"
mod = "MThrustTunedHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_sac_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="tuned",
                 flight_mode=-1)

'''
run_ddpg_training(num_runs=num_runs,
                  total_steps=total_steps,
                  eval_freq=eval_freq,
                  shift=shift,
                  env_id=env_id,
                  policy_kwargs=policy_kwargs,
                  run=run,
                  mod=mod,
                  dir=dir,
                  check_root_dir=check_root_dir,
                  lr_mode="constant",
                  lr=lr,
                  hyperparam_mode="tuned")


run_sac_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="default")


run_sac_training(num_runs=5,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="default")

mod = "TunedHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_sac_training(num_runs=5,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="tuned")
                 
mod = "TunedHyperLinearLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_sac_training(num_runs=5,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="linear",
                 lr=lr,
                 hyperparam_mode="tuned")

mod = "TunedHyperExponentialLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_sac_training(num_runs=5,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="exponential",
                 lr=lr,
                 hyperparam_mode="tuned")

mod = "TunedHyperCosineLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_sac_training(num_runs=5,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="cosine",
                 lr=lr,
                 hyperparam_mode="tuned")


run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="default")

mod = "TunedHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="constant",
                 lr=lr,
                 hyperparam_mode="tuned")

mod = "TunedHyperLinearLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="linear",
                 lr=lr,
                 hyperparam_mode="tuned")

mod = "TunedHyperExponentialLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="exponential",
                 lr=lr,
                 hyperparam_mode="tuned")

mod = "TunedHyperCosineLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

run_ppo_training(num_runs=num_runs,
                 total_steps=total_steps,
                 eval_freq=eval_freq,
                 shift=shift,
                 env_id=env_id,
                 policy_kwargs=policy_kwargs,
                 run=run,
                 mod=mod,
                 dir=dir,
                 check_root_dir=check_root_dir,
                 lr_mode="cosine",
                 lr=lr,
                 hyperparam_mode="tuned")'''
