from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np
# from stable_baselines3.common.evaluation import evaluate_policy
from Evaluation.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


def average_trajectories(trajectories) -> list[list[np.array]]:
    """
    Average the trajectories. Used for plotting the average trajectory over a number of evaluation episodes.
    :param trajectories: List of trajectories
    """

    max_length = max(len(traj) for traj in trajectories)  # Find the longest trajectory
    cum_sums = np.zeros(max_length)  # Initialize cumulative sums
    counts = np.zeros(max_length)  # Initialize counts

    # Accumulate sums and counts
    for traj in trajectories:
        for i, arr in enumerate(traj):
            cum_sums[i] += arr  # Add array value to the cumulative sum
            counts[i] += 1  # Increment count

    # Calculate averages
    averages = cum_sums / counts  # Element-wise division to get the average

    return [averages]


def plot_eval(results, var_name, average=False) -> None:
    """
    Plot the evaluation results.

    :param results: (dict) The evaluation results
    """

    assert len(results.keys()) == 1, 'Only one variable can be plotted at a time.'

    if average:
        len_b4_avg = len(results[var_name])
        results[var_name] = average_trajectories(results[var_name])

    plt.figure(figsize=(8, 6))

    for index, sublist in enumerate(results[var_name]):
        plt.plot(sublist, color="blue")

    plt.legend()

    # Add title and labels
    if average:
        plt.title(f'{var_name} over time (averaged over {len_b4_avg} episodes)')
    else:
        plt.title(f'{var_name} over time')
    plt.xlabel('timestep')
    plt.ylabel(f'{var_name}-value')

    # Show the plot
    plt.show()


def aggregate_eval(model, env, n_eval_episodes, render, var_name, deterministic=True) -> dict[str, list[list[np.array]]]:
    """
    Evaluate the model on the environment for a given number of episodes and aggregate the values of a given variable.
    :param model: PPO Model
    :param env: Vectorized environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param render: Whether to render the environment or not
    :param var_name: Name(s) of the variable(s) to aggregate
    :param deterministic: Whether to use deterministic actions
    :return: Dictionary containing the aggregated values of the variable(s)
    """
    assert type(var_name) == str or type(var_name) == list, 'Variable name must be a string or a list of strings.'

    episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes,
                                                                           render=render, deterministic=deterministic,
                                                                           return_episode_rewards=True)
    res = {}

    if type(var_name) == list:
        for var in var_name:
            res[var] = []
            if var in all_obs[0][0].keys():
                for ep in all_obs:
                    res[var].append([obs[var].squeeze() for obs in ep])
            elif var in all_infos[0][0].keys():
                for ep in all_infos:
                    res[var].append([info[var].squeeze() for info in ep])
            else:
                raise ValueError(f'Variable names not found in observations or infos.')
    else:
        if var_name in ['scaled_smooth_reward', 'scaled_los_reward']:
            raise ValueError(f'Please use the aggregate_reward_eval function to aggregate the reward components.')

        res[var_name] = []
        if var_name in all_obs[0][0].keys():
            for ep in all_obs:
                res[var_name].append([obs[var_name].squeeze() for obs in ep])
        elif var_name in all_infos[0][0].keys():
            for ep in all_infos:
                res[var_name].append([info[var_name].squeeze() for info in ep])
        else:
            raise ValueError(f'Variable name {var_name} not found in observations or infos.')

    return res


def aggregate_reward_eval(model, env, n_eval_episodes, render, reward_name, deterministic=True) -> dict[str, list[list[np.array]]]:
    """
    Evaluate the model on the environment for a given number of episodes and aggregate the values of a given reward component.
    :param model: PPO model
    :param env: Vectorized environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param render: Whether to render the environment or not
    :param reward_name: Name of the reward component to aggregate
    :param deterministic: Whether to use deterministic actions
    :return: Dictionary containing the aggregated values of the reward component
    """

    assert type(reward_name) == str, 'Reward name must be a string.'
    assert reward_name in ['scaled_smooth_reward', 'scaled_los_reward'], 'Reward name must be either "scaled_smooth_reward" or "scaled_los_reward".'

    episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render, deterministic=deterministic, return_episode_rewards=True)
    reward_components = {reward_name: []}

    for ep in all_infos:
        reward_components[reward_name].append([info['reward_components'][reward_name].item() for info in ep])

    return reward_components


def aggregate_smoothness(model, env, n_eval_episodes, render, deterministic=True) -> dict[str, list[list[np.array]]]:
    """
    Evaluate the model on the environment for a given number of episodes and aggregate the values of the smoothness.
    :param model: PPO model
    :param env: Vectorized environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param render: Whether to render the environment or not
    :param deterministic: Whether to use deterministic actions
    :return: Dictionary containing the aggregated values of the smoothness
    """
    result = aggregate_eval(model, env, n_eval_episodes=n_eval_episodes, render=render, var_name=['aux_state'], deterministic=deterministic)
    res = {"smoothness": []}

    for ep in result["aux_state"]:
        res["smoothness"].append([np.linalg.norm(i) for i in ep])

    for ep in res["smoothness"]:  # Remove the last element of each episode to avoid the sudden drop in the plot
        ep.pop(-1)

    return res


def plot_trajectory_with_target(trajectory_points, target):
    """
    Plot a trajectory in 3D space from a list of 3D points, with increasing visibility, and a target point.

    :param trajectory_points: List of numpy arrays, each array is of shape (3,) representing a point in 3D space.
    :param target: A numpy array of shape (3,) representing the target point in 3D space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for plotting
    x, y, z = zip(*trajectory_points)  # Unpack points into separate coordinate lists

    # Plot trajectory with increasing visibility
    for i in range(len(trajectory_points)):
        alpha = i / len(trajectory_points) * 0.9 + 0.1  # Gradually increase visibility
        ax.plot(x[:i + 1], y[:i + 1], z[:i + 1], color='blue', alpha=alpha)

    # Plot the target point
    ax.scatter(target[0], target[1], target[2], color='red', s=100, label='Target')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    plt.show()


def visualize_model(model, env, n_eval_episodes, render, verbose=True):
    """
    Visualize the model's performance on the environment for a given number of episodes.
    :param model: PPO model
    :param env: Vectorized environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param deterministic: Whether to use deterministic actions
    :param render_mode: Render mode
    :param var_name: Name of the variable to visualize
    """

    episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render, deterministic=True, return_episode_rewards=True)

    if verbose:
        return episode_rewards, episode_lengths, all_obs, all_infos


#  ---------------------------------------------------------------------------------------------------------------------

model_path = "./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted-AngVel/best_model"
env_id = "SingleWaypointQuadXEnv-v0"

render = False
num_eval_eps = 5

# Create model and environment
env = gym.make(env_id, render_mode="human" if render else None)
_ = env.reset()
model = PPO.load(model_path, deterministic=True)

# OBS: dict_keys(['a_azimuth_angle', 'a_elevation_angle', 'altitude', 'ang_vel', 'aux_state', 't_azimuth_angle', 't_elevation_angle'])
# INFO: dict_keys(['out_of_bounds', 'collision', 'env_complete', 'num_targets_reached', 'non_observables', 'reward_components', 'TimeLimit.truncated'])


print("---------------------------- evaluate_policy --------------------------------")

# visualize_model(model, env, num_eval_eps, render, verbose=False)
# result = aggregate_eval(model, env, num_eval_eps, render, var_name="scaled_los_reward")
# result = aggregate_reward_eval(model, env, num_eval_eps, render, "scaled_los_reward")
# plot_eval(result, "scaled_los_reward", average=True)
# smoothness = aggregate_smoothness(model, env, num_eval_eps, render)



print("---------------------------- evaluate_policy --------------------------------")

#  ---------------------------------------------------------------------------------------------------------------------
