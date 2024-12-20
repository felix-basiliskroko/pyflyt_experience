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
from stable_baselines3.common.callbacks import EvalCallback
from Evaluation.vis_model import aggregate_eval

# Logdir
eval_freq = 25_000

# For Mac-Machine:
# log_root_dir = "./logs/final_log/StaticWaypointEnv"
# check_root_dir = "./checkpoints/StaticWaypointEnv"

# For Windows-Machine:
log_root_dir = "../logs/final_log/StaticWaypointEnv"
check_root_dir = "../checkpoints/StaticWaypointEnv"
num_runs = 1
steep_grad = 1.0
eval_eps = 500
eval_during_training = 50
neg_rew = False

run = "SingleWaypointNavigation"
mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}"
dir = f'{log_root_dir}/{run}/{mod}'

best_model_path = None
best_num_env_complete = -1
all_results = []

for experiment in range(7):
    for i in range(num_runs):
        env_id = "SingleWaypointQuadXEnv-v0"
        vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"negative_reward": neg_rew, "steep_grad": steep_grad})
        policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

        eval_env = make_vec_env(env_id=env_id, seed=42, env_kwargs={"negative_reward": neg_rew, "steep_grad": steep_grad})
        eval_callback = EvalCallback(eval_env, best_model_save_path=f"./{check_root_dir}/{run}/{mod}/best_model_run_{i}",
                                     log_path=f"./{check_root_dir}/{run}/{mod}", eval_freq=eval_freq,
                                     deterministic=True, render=True, n_eval_episodes=100)
        device = "cuda" if t.cuda.is_available() else "cpu"

        lr = 7e-3
        if experiment == 4:
            mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}/LR_Schedule=linear"
            dir = f'{log_root_dir}/{run}/{mod}'
            lr = linear_schedule(initial_lr=lr)
        elif experiment == 5:
            mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}/LR_Schedule=exp(0.99)"
            dir = f'{log_root_dir}/{run}/{mod}'
            lr = exponential_schedule(initial_lr=lr, decay_rate=0.99)
        elif experiment == 6:
            mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}/LR_Schedule=cosine(min=1e-4)"
            lr = cosine_annealing_schedule(initial_lr=lr, min_lr=1e-3)

        model = PPO("MultiInputPolicy", vec_env, verbose=0, tensorboard_log=dir, policy_kwargs=policy_kwargs,
                    ent_coef=0.0,
                    gamma=0.985,
                    learning_rate=lr,
                    gae_lambda=0.9875,
                    batch_size=1024,
                    device=device)  # For non-dict observation space
        model.learn(total_timesteps=250_000, callback=eval_callback, tb_log_name=f'run_{i}')

        # Evaluate the model and check for best num_env_complete
        model_path = f"./{check_root_dir}/{run}/{mod}/best_model_run_{i}/best_model.zip"
        eval_env = gym.make(env_id, render_mode=None)
        _ = eval_env.reset()
        model = PPO.load(model_path, deterministic=True)

        result = aggregate_eval(model, eval_env, num_eval_eps=500, render=False, include_waypoints=True)
        all_results.append((model_path, result["num_term_flags"]))
        if result["num_term_flags"]["num_env_complete"] > best_num_env_complete:
            best_num_env_complete = result["num_term_flags"]["num_env_complete"]
            best_model_path = model_path

    # Write the results to the results file
    with open("results.txt", "a") as f:
        if experiment < 4:
            f.write(f"\nExperiment: Steep Grad = {steep_grad}, Negative Reward = {neg_rew}\n")
        if experiment == 4:
            f.write("\n\nExperiment: Linear Learning Rate Decay\n")
        elif experiment == 5:
            f.write("\n\nExperiment: Exponential Learning Rate Decay\n")
        elif experiment == 6:
            f.write("\n\nExperiment: Cosine Annealing Learning Rate Decay\n")
        f.write(f"Best model checkpoint: {best_model_path}\n")
        for path, flags in all_results:
            f.write(f"Model: {path}\n")
            for key, value in flags.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    if experiment == 0:
        steep_grad = 1.0
        neg_rew = True
        mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}"
        dir = f'{log_root_dir}/{run}/{mod}'

    if experiment == 1:
        steep_grad = 2.0
        neg_rew = False
        mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}"
        dir = f'{log_root_dir}/{run}/{mod}'

    if experiment == 2:
        steep_grad = 2.0
        neg_rew = True
        mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}"
        dir = f'{log_root_dir}/{run}/{mod}'

    elif experiment == 3:  # Find best permutation of negative reward and steep grad
        with open("results.txt", "r") as f:
            lines = f.readlines()

        best_result = None
        best_config = None
        for line in lines:
            if "Best model checkpoint" in line:
                checkpoint_line = line
            if "Steep Grad" in line and "Negative Reward" in line:
                config_line = line
                if best_result is None or best_num_env_complete > best_result:
                    best_result = best_num_env_complete
                    best_config = config_line

        # Parse the best configuration and update variables
        if best_config:
            steep_grad = float(best_config.split("Steep Grad = ")[1].split(",")[0])
            neg_rew = "True" in best_config.split("Negative Reward = ")[1]
            mod = f"E1_LOSOnly_SteepGrad={steep_grad}_NegReward={neg_rew}"
            dir = f'{log_root_dir}/{run}/{mod}'
