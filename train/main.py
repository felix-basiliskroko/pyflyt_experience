import argparse
import sys
import torch as t
import os

from train_ppo import run_ppo_training
from train_sac import run_sac_training
from train_ddpg import run_ddpg_training


def parse_args():
    parser = argparse.ArgumentParser(description="Run RL training for UAV navigation.")

    parser.add_argument("--algorithm", type=str, choices=["ppo", "sac", "ddpg"], required=True, help="Choose the RL algorithm.")
    parser.add_argument("--num_runs", type=int, default=1, help="Number of training runs.")
    parser.add_argument("--total_steps", type=int, required=True, help="Total training steps.")
    parser.add_argument("--eval_freq", type=int, default=20_000, help="Evaluation frequency.")
    parser.add_argument("--shift", type=float, default=0.75, help="Shift applied to reward function during training.")

    # Set default learning rates based on algorithm choice
    default_lr = {"ppo": 3e-4, "sac": 3e-4, "ddpg": 7e-5}
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")

    parser.add_argument("--lr_mode", type=str, choices=["constant", "linear", "exponential", "cosine"], default="constant", help="Learning rate schedule.")
    parser.add_argument("--hyperparam_mode", type=str, choices=["default", "tuned"], required=True, help="Hyperparameter configuration.")
    parser.add_argument("--flight_mode", type=int, choices=[1, -1], required=True, help="Flight mode setting.")
    parser.add_argument("--log_root_dir", type=str, default="../logs/new_runs", help="Root directory where tensorboard logs will be saved.")
    parser.add_argument("--check_root_dir", type=str, default="../checkpoints/new_runs", help="Root directory where the model checkpoints will be saved.")
    parser.add_argument("--run_name", type=str, required=True, help="Name of the run.")

    args = parser.parse_args()

    # Assign default learning rate if not specified
    if args.lr is None:
        args.lr = default_lr[args.algorithm]

    return args


def main():
    sys.path.append(os.path.abspath("Envs"))  # Adjust the path accordingly
    args = parse_args()
    log_root_dir, check_root_dir = args.log_root_dir, args.check_root_dir
    run_name = f'{args.algorithm.upper()}_{"angular" if args.flight_mode == 1 else "thrust"}_{args.run_name}'
    mod = f'{args.algorithm.upper()}_hyperparam={args.hyperparam_mode}_shift={args.shift}_flight_mode={"angular" if args.flight_mode == 1 else "thrust"}_lr_mode={args.lr_mode}'

    os.makedirs(log_root_dir, exist_ok=True)
    os.makedirs(check_root_dir, exist_ok=True)

    # Define policy architecture
    if args.algorithm == "ppo":
        policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    else:  # SAC/DDPG
        policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))

    # Run training
    if args.algorithm == "ppo":
        run_ppo_training(num_runs=args.num_runs,
                         total_steps=args.total_steps,
                         eval_freq=args.eval_freq,
                         shift=args.shift,
                         env_id="SingleWaypointQuadXEnv-v0",
                         policy_kwargs=policy_kwargs,
                         run=run_name,
                         mod=mod,
                         dir=log_root_dir,
                         check_root_dir=check_root_dir,
                         lr_mode=args.lr_mode,
                         lr=args.lr,
                         hyperparam_mode=args.hyperparam_mode,
                         flight_mode=args.flight_mode)

    elif args.algorithm == "sac":
        run_sac_training(
            num_runs=args.num_runs,
            total_steps=args.total_steps,
            eval_freq=args.eval_freq,
            shift=args.shift,
            env_id="SingleWaypointQuadXEnv-v0",
            policy_kwargs=policy_kwargs,
            run=run_name,
            mod=mod,
            dir=log_root_dir,
            check_root_dir=check_root_dir,
            lr_mode=args.lr_mode,
            lr=args.lr,
            hyperparam_mode=args.hyperparam_mode,
            flight_mode=args.flight_mode
        )

    elif args.algorithm == "ddpg":
        run_ddpg_training(
            num_runs=args.num_runs,
            total_steps=args.total_steps,
            eval_freq=args.eval_freq,
            shift=args.shift,
            env_id="SingleWaypointQuadXEnv-v0",
            policy_kwargs=policy_kwargs,
            run=run_name,
            mod=mod,
            dir=log_root_dir,
            check_root_dir=check_root_dir,
            lr_mode=args.lr_mode,
            lr=args.lr,
            hyperparam_mode=args.hyperparam_mode,
            flight_mode=args.flight_mode
        )


if __name__ == "__main__":
    main()

# python3 main.py --algorithm="ppo" --total_steps=5000 --eval_freq=500 --hyperparam_mode="default" --flight_mode=1 --run_name="CheckItOut"
