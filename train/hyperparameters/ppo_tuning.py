import json
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import torch as t
import time

from Envs import register
from Evaluation.custom_eval_callback import CustomEvalCallback
from Evaluation.vis_model import aggregate_eval


def ppo_tune(env_id: str, log_dir: str, buckets: int, num_total_steps: int, num_buffer_steps: int, n_runs: int,
             n_envs: int, n_eval_eps: int, hyperparameter_save_path: str,
             ent_coeffs: list, gammas: list, gae_lambdas: list, learning_rates: list):
    """
    Tune the PPO hyperparameters using a grid search approach.
    :param env_id: Environment id
    :param log_dir: Directory to save the logs
    :param buckets: Number of buckets to divide the hyperparameter range into
    :param num_total_steps: Total number of steps to train the model
    :param num_buffer_steps: Number of steps to collect before updating the model
    :param n_runs: Number of runs per hyperparameter permutation
    :param n_envs: Number of environments to run in parallel
    :param n_eval_eps: Number of episodes to evaluate the model on
    :param hyperparameter_save_path: Path to save the hyperparameter results
    :param ent_coeffs:
    :param gammas:
    :param gae_lambdas:
    :param learning_rates:
    :return:
    """

    device = "cuda" if t.cuda.is_available() else "cpu"
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))

    # Create the hyperparameter grid
    ent_coeffs = np.linspace(ent_coeffs[0], ent_coeffs[1], buckets)
    gammas = np.linspace(gammas[0], gammas[1], buckets)
    gae_lambdas = np.linspace(gae_lambdas[0], gae_lambdas[1], buckets)
    learning_rates = np.linspace(learning_rates[0], learning_rates[1], buckets)

    total_num_runs = len(ent_coeffs) * len(gammas) * len(gae_lambdas) * len(learning_rates) * n_runs
    train_env = make_vec_env(env_id=env_id, n_envs=n_envs, seed=69)
    eval_env = make_vec_env(env_id=env_id, seed=42)

    num_completed_runs = 0


    hyperparameter_results = {}
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for ent in ent_coeffs:
        for gamma in gammas:
            for gae in gae_lambdas:
                for lr in learning_rates:
                    hyper_key = f"ent={ent}_gamma={gamma}_gae={gae}_lr={lr}"
                    hyperparameter_results[hyper_key] = {}
                    start_time = time.time()
                    for run_id in range(n_runs):
                        run_dir = f"{log_dir}/ent={ent}_gamma={gamma}_gae={gae}_lr={lr}"
                        eval_callback = CustomEvalCallback(eval_env, best_model_save_path=log_dir,
                                                     log_path=None, eval_freq=int(num_total_steps/3),
                                                     deterministic=True, render=False, n_eval_episodes=5)

                        model = PPO("MultiInputPolicy",
                                    train_env,
                                    verbose=0,
                                    tensorboard_log=run_dir,
                                    policy_kwargs=policy_kwargs,
                                    ent_coef=ent,
                                    gamma=gamma,
                                    gae_lambda=gae,
                                    learning_rate=lr,
                                    n_steps=num_buffer_steps,
                                    batch_size=num_buffer_steps,
                                    device=device)

                        model.learn(total_timesteps=num_total_steps, callback=eval_callback, tb_log_name="run")

                        # Evaluate the model
                        env = gym.make(env_id, render_mode=None)
                        _ = env.reset()
                        eval_model = PPO.load(f"{log_dir}/best_model", deterministic=True)
                        result = aggregate_eval(model=eval_model, env=env, n_eval_episodes=n_eval_eps, render=False, include_waypoints=True)
                        # result = serialize(result)  # Convert numpy arrays to lists for JSON serialization

                        # Load the existing JSON file
                        with open(hyperparameter_save_path, "r") as f:
                            hyperparameter_results = json.load(f)

                        # Update the results
                        if hyper_key not in hyperparameter_results:
                            hyperparameter_results[hyper_key] = {}
                        hyperparameter_results[hyper_key][str(run_id)] = result

                        # Save the updated results back to the file
                        with open(hyperparameter_save_path, "w") as f:
                            json.dump(hyperparameter_results, f, indent=4)

                        num_completed_runs += 1

                    # Calculate averages and standard deviations after n_runs
                    unstable, collisions, out_of_bounds = [], [], []
                    for run_id in range(n_runs):
                        run_data = hyperparameter_results[hyper_key][str(run_id)]
                        unstable.append(run_data["num_term_flags"]["num_unstable"])
                        collisions.append(run_data["num_term_flags"]["num_collision"])
                        out_of_bounds.append(run_data["num_term_flags"]["num_out_of_bounds"])

                    # Update the hyperparameter results
                    hyperparameter_results[hyper_key]["avg_num_unstable"] = np.mean(unstable)
                    hyperparameter_results[hyper_key]["std_num_unstable"] = np.std(unstable)
                    hyperparameter_results[hyper_key]["avg_num_collision"] = np.mean(collisions)
                    hyperparameter_results[hyper_key]["std_num_collision"] = np.std(collisions)
                    hyperparameter_results[hyper_key]["avg_num_out_of_bounds"] = np.mean(out_of_bounds)
                    hyperparameter_results[hyper_key]["std_num_out_of_bounds"] = np.std(out_of_bounds)

                    with open(hyperparameter_save_path, "w") as f:
                        json.dump(hyperparameter_results, f, indent=4)

                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    print("# -----------------------------------------------")
                    print(f'Complete permutation: {hyper_key}')
                    print(f'Completed runs: {(num_completed_runs/total_num_runs) * 100:.2f}%')

                    # Calculate elapsed time and format it
                    if elapsed_time < 60:
                        print(f"Execution time: {elapsed_time:.2f} seconds")
                    elif elapsed_time < 3600:
                        minutes, seconds = divmod(elapsed_time, 60)
                        print(f"Execution time: {int(minutes)} minutes and {seconds:.2f} seconds")
                    else:
                        hours, remainder = divmod(elapsed_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"Execution time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")

                    # Calculate estimated remaining time
                    est_rem_time = (total_num_runs - num_completed_runs) * elapsed_time

                    # Format estimated remaining time dynamically
                    if est_rem_time < 60:
                        print(f"Estimated remaining time: {est_rem_time:.2f} seconds")
                    elif est_rem_time < 3600:
                        minutes, seconds = divmod(est_rem_time, 60)
                        print(f"Estimated remaining time: {int(minutes)} minutes and {seconds:.2f} seconds")
                    elif est_rem_time < 86400:  # Less than a day
                        hours, remainder = divmod(est_rem_time, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"Estimated remaining time: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
                    else:  # More than a day
                        days, remainder = divmod(est_rem_time, 86400)
                        hours, remainder = divmod(remainder, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        print(f"Estimated remaining time: {int(days)} days, {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")


def batch_tune(env_id: str, log_dir: str, values: list, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69, env_kwargs={"reward_shift": 0.75, "steep_grad": 1.0})
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, "batch_size")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    for b_size in values:
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    batch_size=b_size,
                    device=device)
        model.learn(total_timesteps=num_steps, tb_log_name=f'batch_size={b_size}', callback=eval_callback)
    print("Finished tuning entropy coefficient.")


def gamma_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, "gamma")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    gamma_vals = np.linspace(value_range[0], value_range[1], buckets)

    for gamma in gamma_vals:
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    gamma=gamma,
                    device=device)
        model.learn(total_timesteps=num_steps, callback=eval_callback, tb_log_name=f'gamma={round(gamma, 3)}')
    print("Finished tuning gamma.")


def entr_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, "entropy_coeff")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    ent_coeff_vals = np.linspace(value_range[0], value_range[1], buckets)

    for ent_coeff in ent_coeff_vals:
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    gamma=0.975,
                    ent_coef=ent_coeff,
                    device=device)
        model.learn(total_timesteps=num_steps, tb_log_name=f'ent_coeff={round(ent_coeff, 3)}', callback=eval_callback)
    print("Finished tuning entropy coefficient.")


def gae_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                        log_path=None, eval_freq=eval_freq,
                                        deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, "gae")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    gae_vals = np.linspace(value_range[0], value_range[1], buckets)

    for gae in gae_vals:
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    gamma=0.975,
                    ent_coef=0.009,
                    gae_lambda=gae,
                    device=device)
        model.learn(total_timesteps=num_steps, callback=eval_callback, tb_log_name=f'gae={round(gae, 3)}')
    print("Finished tuning gae_lambda.")


def lr_tune(env_id: str, log_dir: str, value_range: list, buckets: int, num_steps: int):
    device = "cuda" if t.cuda.is_available() else "cpu"
    eval_freq = 20_000

    vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=69)
    eval_env = gym.make(env_id, reward_shift=0.75, steep_grad=1.0)
    eval_callback = CustomEvalCallback(eval_env, best_model_save_path=None,
                                       log_path=None, eval_freq=eval_freq,
                                       deterministic=True, render=True, n_eval_episodes=100)

    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
    log_dir = os.path.join(log_dir, "lr")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    lr_vals = np.linspace(value_range[0], value_range[1], buckets)

    for lr in lr_vals:
        model = PPO("MultiInputPolicy",
                    vec_env,
                    verbose=0,
                    tensorboard_log=log_dir,
                    policy_kwargs=policy_kwargs,
                    batch_size=256,
                    ent_coef=0.009,
                    gamma=0.975,
                    gae_lambda=0.94,
                    learning_rate=lr,
                    device=device)
        model.learn(total_timesteps=num_steps, callback=eval_callback, tb_log_name=f'lr={round(lr, 5)}')
    print("Finished tuning the learning rate.")