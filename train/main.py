from train_ppo import run_ppo_training
import torch as t

# Logdir
num_runs = 3
total_steps = 600_000
eval_freq = 20_000
shift = 0.75
# shifts = [0.0, 0.25, 0.5, 0.75, 1.0]

# For Mac-Machine:
log_root_dir = "../logs/tensorboard_log/Final/"
check_root_dir = "../checkpoints/Final"

# For Windows-Machine:
# log_root_dir = "../logs/tensorboard_log/StaticWaypointEnv"
# check_root_dir = "../checkpoints/StaticWaypointEnv/final"

run = "SingleWaypointNavigation"
mod = "DefaultHyperConstantLearningRate"
dir = f'{log_root_dir}/{run}/{mod}'

env_id = "SingleWaypointQuadXEnv-v0"
policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
lr = 9e-4

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
                 hyperparam_mode="tuned")