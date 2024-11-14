from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
# from stable_baselines3.common.evaluation import evaluate_policy
from Evaluation.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


def plotly_vector_field(linear_positions, linear_velocities, target_vector, size=1, save_path=None):
    """
    Plots the trajectory of the drone in 3D space with its corresponding velocity vectors as well as target.
    :param linear_positions: list of sampled linear positions
    :param linear_velocities: list of sampled linear velocities
    :param target_vector: target vector
    :param size: size of the cones, need adjustment based on the magnitude of the velocities
    """
    linear_velocity = np.concatenate((linear_positions, linear_velocities), axis=1)

    x, y, z, u, v, w = linear_velocity.T

    pl_ice = [
        [0.0, 'rgb(3, 5, 18)'],
        [0.11, 'rgb(27, 26, 54)'],
        [0.22, 'rgb(48, 46, 95)'],
        [0.33, 'rgb(60, 66, 136)'],
        [0.44, 'rgb(62, 93, 168)'],
        [0.56, 'rgb(66, 122, 183)'],
        [0.67, 'rgb(82, 149, 192)'],
        [0.78, 'rgb(106, 177, 203)'],
        [0.89, 'rgb(140, 203, 219)'],
        [1.0, 'rgb(188, 227, 235)']
    ]

    trace1 = go.Cone(
        x=x, y=y, z=z,
        u=u, v=v, w=w,
        colorscale=pl_ice,
        sizemode='absolute',
        sizeref=size,
        colorbar=dict(thickness=40, ticklen=4),
        anchor='tip'
    )

    trace2 = go.Scatter3d(
        x=[target_vector[0]],
        y=[target_vector[1]],
        z=[target_vector[2]],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            opacity=0.8
        )
    )

    layout = go.Layout(
        width=900,
        height=750,
        autosize=False,
        scene=dict(
            camera=dict(eye=dict(x=1.2, y=1.2, z=0.6)),
            xaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            aspectratio=dict(x=1, y=1, z=0.8)
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    if save_path:
        fig.write_html(save_path)
    else:
        # warnings.warn("The plot can only be displayed when the function is called in a Jupyter notebook.")
        fig.show()


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


def plot_eval(results, average=False) -> None:
    """
    Plot the evaluation results.

    :param results: (dict) The evaluation results
    """

    var_name = next(iter(results.keys()))
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


def plot_multiple_eval(results, average=False) -> None:
    """
    Plot the evaluation results for multiple variables.

    :param results: (dict) The evaluation results where keys are variable names
    """

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))  # Generates a color map

    plt.figure(figsize=(10, 8))
    legend_labels = []

    for (var_name, sublist), color in zip(results.items(), colors):
        if average:
            len_b4_avg = len(sublist)
            sublist = average_trajectories(sublist)  # Make sure this returns a flat list of numbers

        for data in sublist:
            if isinstance(data, np.ndarray):
                data = data.ravel()  # Converts array to 1D, if needed
            plt.plot(data, color=color)

        label = f'{var_name} (avg over {len_b4_avg} episodes)' if average else var_name
        legend_labels.append((label, color))

    # Create a custom legend
    for label, color in legend_labels:
        plt.plot([], color=color, label=label)

    plt.legend(title="Variable")
    plt.title('Evaluation Results Over Time')
    plt.xlabel('Timestep')
    plt.ylabel('Value')
    plt.show()


def aggregate_eval(model, env, n_eval_episodes, render, deterministic=True, include_waypoints=True) -> dict[str, list[list[np.array]]]:
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

    episode_rewards, episode_lengths, all_obs, all_infos, waypoints = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes,
                                                                           render=render, deterministic=deterministic,
                                                                           return_episode_rewards=True)
    res = {}
    var_name = ["azimuth_angle", "elevation_angle", "aux_state",
                "ang_vel", "altitude", "angular_position", "quaternion",
                "linear_position", "linear_velocity", "distance_to_target"]

    for var in var_name:
        res[var] = []
        if var in all_obs[0][0].keys():
            for ep in all_obs:
                try:
                    res[var].append([obs[var].squeeze() for obs in ep])
                except AttributeError:  # If the variable is not an array
                    res[var].append([obs[var] for obs in ep])
                res[var][-1].pop(-1)
        elif var in all_infos[0][0].keys():
            for ep in all_infos:
                try:
                    res[var].append([info[var].squeeze() for info in ep])
                except AttributeError:
                    res[var].append([info[var] for info in ep])
                res[var][-1].pop(-1)
        else:
            raise ValueError(f'Variable "{var}" not found in observations or infos.')

    # Include waypoints in the results
    res["waypoints"] = waypoints

    # Calculate smoothness of the control inputs and the thrust-level
    res["smoothness"] = []
    res["thrust"] = []

    for ep in res["aux_state"]:
        res["smoothness"].append([np.linalg.norm(i) for i in ep])
        res["thrust"].append([i[3] for i in ep])

    for ep in res["smoothness"]:  # Remove the last element of each episode to avoid the sudden drop in the plot
        ep.pop(-1)

    for ep in res["thrust"]:  # Remove the last element of each episode to avoid the sudden drop in the plot
        ep.pop(-1)

    return res


def visualize_plotly_model(result, model, env, n_eval_episodes, save_path=None):
    data = []
    for episode_index, (positions, waypoint) in enumerate(zip(result['linear_position'], result['waypoints'])):
        for pos in positions:
            data.append({
                'Episode': episode_index,
                'X': pos[0],
                'Y': pos[1],
                'Z': pos[2],
                'Type': 'Position'
            })
        # Adding waypoint for the episode
        data.append({
            'Episode': episode_index,
            'X': waypoint[0],
            'Y': waypoint[1],
            'Z': waypoint[2],
            'Type': 'Waypoint'
        })

    df = pd.DataFrame(data)

    # Plotting using Plotly Express
    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Episode', symbol='Type', title='Trajectory and Waypoints per Episode')
    fig.update_traces(marker=dict(size=5))

    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()

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
