import math

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO, SAC
import gymnasium as gym
from tqdm import tqdm

from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import glob
import os
from tensorboard.backend.event_processing import event_accumulator
from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.evaluation import evaluate_policy
from Evaluation.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


def plotly_vector_field(linear_positions, linear_velocities, target_vector, size=1, save_path=None, camera_angle=None):
    """
    Plots the trajectory of the drone in 3D space with its corresponding velocity vectors as well as the target.
    :param linear_positions: list of sampled linear positions
    :param linear_velocities: list of sampled linear velocities
    :param target_vector: target vector
    :param size: size of the cones, need adjustment based on the magnitude of the velocities
    :param save_path: path to save the output (excluding extension)
    :param camera_angle: dictionary specifying camera perspective (e.g., {"eye": {"x": 1.2, "y": 1.2, "z": 0.6}})
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
        anchor='tip',
        showscale=False
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
            camera=camera_angle or dict(eye=dict(x=1.2, y=1.2, z=0.6)),
            xaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgb(235, 235, 235)", gridcolor="rgb(255, 255, 255)", zerolinecolor="rgb(255, 255, 255)"),
            aspectratio=dict(x=1, y=1, z=0.8)
        )
    )

    fig = go.Figure(data=[trace1, trace2], layout=layout)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(f"{save_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        fig.write_html(f"{save_path}.html")
        print(f"Figure saved at {save_path}.pdf")
    else:
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
    Plot the evaluation results using Plotly.

    :param results: (dict) The evaluation results
    """

    var_name = next(iter(results.keys()))
    assert len(results.keys()) == 1, 'Only one variable can be plotted at a time.'

    if average:
        len_b4_avg = len(results[var_name])
        results[var_name] = average_trajectories(results[var_name])

    fig = go.Figure()

    for index, sublist in enumerate(results[var_name]):
        fig.add_trace(go.Scatter(y=sublist, line=dict(color="blue"), name=var_name))

    # Set the title based on whether averaging is done
    title_text = f'{var_name} over time (averaged over {len_b4_avg} episodes)' if average else f'{var_name} over time'

    # Set layout options
    fig.update_layout(
        title=title_text,
        xaxis_title='Timestep',
        yaxis_title=f'{var_name}-value',
        legend_title=var_name,
        # Adjust figure size similar to matplotlib
        autosize=False,
        width=800,
        height=600
    )

    fig.show()


def plot_multiple_eval(results, average=False, title="Results", save_path=None) -> None:
    """
    Plot the evaluation results for multiple variables using Plotly.

    :param average: (bool) Whether to average the trajectories or not
    :param save_path: (str) Path to save the figure
    :param title: (str) Title of the plot
    :param results: (dict) The evaluation results where keys are variable names
    """
    # Generates a color map using Plotly's default sequence
    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    for (var_name, sublist), color in zip(results.items(), colors):
        if average:
            len_b4_avg = len(sublist)
            sublist = average_trajectories(sublist)  # Ensure this returns a flat list of numbers

        for data in sublist:
            if isinstance(data, np.ndarray):
                data = data.ravel()  # Converts array to 1D, if needed
            fig.add_trace(go.Scatter(y=data, line=dict(color=color), name=var_name))

        # Update the label if averaging was performed
        if average:
            label = f'{var_name} (avg over {len_b4_avg} episodes)'
            # Update the last trace added with the new name
            fig.data[-1].update(name=label)

    # Set layout options
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title='Timestep',
        yaxis_title='Value',
        legend_title='Variable',
        # Make legend horizontal
        legend=dict(orientation="h"),
        # Adjust figure size similar to matplotlib
        autosize=False,
        width=1000,
        height=800
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(f"{save_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        print(f"Figure saved at {save_path}.pdf")

    fig.show()


def plot_termination_flags(flag_data, save_path=None):
    labels = list(flag_data.keys())
    values = list(flag_data.values())

    total = sum(values)
    relative_frequencies = [(value / total) * 100 for value in values]

    # Create a plot
    fig = go.Figure(data=[go.Bar(
        x=labels,
        y=relative_frequencies,
        text=values,
        hoverinfo='text+y',
        textposition='auto',
    )])

    fig.update_layout(
        title='Termination Flags Distribution',
        xaxis_title='Termination Flags',
        yaxis_title='Relative Frequency (%)',
        hovermode='closest'
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(f"{save_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        print(f"Figure saved at {save_path}.pdf")

    fig.show()


def aggregate_eval(model, env, n_eval_episodes, render, deterministic=True, include_waypoints=True):
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
    var_name = ["azimuth_angle", "elevation_angle",
                "ang_vel", "altitude", "ang_pos", "quaternion", "aux_state",
                "linear_position", "linear_velocity", "distance_to_target", "action",
                "unstable", "collision", "out_of_bounds", "env_complete", "m_thrusts"]

    for var in var_name:
        res[var] = []
        if var in all_obs[0][0].keys():
            for ep in all_obs:
                try:
                    res[var].append([obs[var].squeeze() for obs in ep])
                except AttributeError:  # If the variable is not an array
                    res[var].append([obs[var] for obs in ep])
                # res[var][-1].pop(-1)
        elif var in all_infos[0][0].keys():
            for ep in all_infos:
                try:
                    res[var].append([info[var].squeeze() for info in ep])
                except AttributeError:
                    res[var].append([info[var] for info in ep])
                # res[var][-1].pop(-1)
        else:
            continue

    # Include waypoints in the results
    res["waypoints"] = waypoints

    # Calculate smoothness of the control inputs and the thrust-level
    res["smoothness"] = []
    res["pitch"], res["yaw"], res["roll"] = [], [], []
    res["pitch_ang"], res["yaw_ang"], res["roll_ang"] = [], [], []
    res["translation_accuracy"] = []

    # Calculate the pitch, yaw, and roll angles
    for ep in res["ang_pos"]:
        res["pitch"].append([i[0] for i in ep[:-1]])
        res["yaw"].append([i[1] for i in ep[:-1]])
        res["roll"].append([i[2] for i in ep[:-1]])

    # Calculate the pitch, yaw, and roll angular velocities
    for ep in res["ang_vel"]:  # ignore the last element of each episode
        res["pitch_ang"].append([i[0] for i in ep[:-1]])
        res["yaw_ang"].append([i[1] for i in ep[:-1]])
        res["roll_ang"].append([i[2] for i in ep[:-1]])

    # Calculate translation accuracy (how accurately p,y,r actions are translated to angular positions of the UAV)
    for angular_list, action_list in zip(res["ang_pos"], res["action"]):
        temp_list = []
        for angular, action in zip(angular_list, action_list):
            diff = np.abs(action[:3] - angular[:3])
            temp_list.append(diff)

        res["translation_accuracy"].append(temp_list)

    for ep in res["ang_pos"]:
        res["smoothness"].append([np.linalg.norm(i) for i in ep[:-1]])

    res["num_term_flags"] = {
        "num_unstable": 0,
        "num_collision": 0,
        "num_out_of_bounds": 0,
        "num_env_complete": 0,
        "out_of_time": 0,
    }

    ep_legnths = {
        "env_complete": [],
        "out_of_time": [],
        "collision": [],
        "unstable": [],
    }

    for ep in res["azimuth_angle"]:
        ep.pop(-1)

    for ep in res["elevation_angle"]:
        ep.pop(-1)

    res["thrust"] = []
    for ep in res["aux_state"]:
        res["thrust"].append([i[3] for i in ep[:-1]])

    for ep in res["unstable"]:
        if ep[-1]:
            res["num_term_flags"]["num_unstable"] += 1
            ep_legnths["unstable"].append(len(ep))

    for ep in res["collision"]:
        if ep[-1]:
            res["num_term_flags"]["num_collision"] += 1
            ep_legnths["collision"].append(len(ep))

    for ep in res["out_of_bounds"]:
        if ep[-1]:
            res["num_term_flags"]["num_out_of_bounds"] += 1
            ep_legnths["out_of_time"].append(len(ep))

    for ep in res["env_complete"]:
        if ep[-1]:
            res["num_term_flags"]["num_env_complete"] += 1
            ep_legnths["env_complete"].append(len(ep))

    res["num_term_flags"]["num_out_of_time"] = n_eval_episodes - (res["num_term_flags"]["num_unstable"]
                                                                  + res["num_term_flags"]["num_collision"]
                                                                  + res["num_term_flags"]["num_out_of_bounds"]
                                                                  + res["num_term_flags"]["num_env_complete"])

    res["termination_lengths"] = {
        "env_complete_mean": np.mean(ep_legnths["env_complete"]),
        "env_complete_std": np.std(ep_legnths["env_complete"]),
        "out_of_time_mean": np.mean(ep_legnths["out_of_time"]),
        "out_of_time_std": np.std(ep_legnths["out_of_time"]),
        "collision_mean": np.mean(ep_legnths["collision"]),
        "collision_std": np.std(ep_legnths["collision"]),
    }

    return res, episode_rewards, episode_lengths


def visualize_plotly_model(result, model, env, n_eval_episodes, save_path=None):
    data = []
    for episode_index, (positions, waypoint) in enumerate(zip(result['linear_position'], result['waypoints'])):
        for step_index, pos in enumerate(positions):
            data.append({
                'Episode': episode_index,
                'Step': step_index,  # Adding step number
                'X': pos[0],
                'Y': pos[1],
                'Z': pos[2],
                'Type': 'Position',
                'Text': f'Step: {step_index}'  # Use 'Text' to store hover info
            })
        # Adding waypoint for the episode
        data.append({
            'Episode': episode_index,
            'Step': len(positions),  # Waypoint is the last step
            'X': waypoint[0],
            'Y': waypoint[1],
            'Z': waypoint[2],
            'Type': 'Waypoint',
            'Text': f'Waypoint, Step: {len(positions)}'
        })

    df = pd.DataFrame(data)

    # Plotting using Plotly Express
    fig = px.scatter_3d(df, x='X', y='Y', z='Z', color='Episode', symbol='Type',
                        title='Trajectory and Waypoints per Episode',
                        hover_data=['Step', 'Text'])  # Adding hover data

    fig.update_traces(marker=dict(size=5))
    fig.update_traces(hovertemplate="Episode: %{customdata[0]}<br>Step: %{customdata[1]}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<br>%{customdata[2]}<extra></extra>")

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

    episode_rewards, episode_lengths, all_obs, all_infos, target_waypoints = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, render=render, deterministic=True, return_episode_rewards=True)

    if verbose:
        return episode_rewards, episode_lengths, all_obs, all_infos


def id_nav_failures(model: BaseAlgorithm, num_eps, save_path=None) -> None:
    import os
    import numpy as np
    import plotly.graph_objects as go

    if hasattr(model.env, 'envs'):  # Check if it's vectorized
        env = model.env.envs[0]  # Unvectorized environment
    elif hasattr(model.env, 'unwrapped'):  # If directly unwrapped
        env = model.env.unwrapped
    else:
        raise ValueError("Environment not found")

    waypoint_heights = []
    reached = []
    failed = []
    reached_episode_lengths = []

    for e in range(num_eps):
        # Reset environment and model
        obs, _ = env.reset()
        term, trunc = False, False
        steps = 0

        wp = env.waypoints.targets[0]  # Get the target waypoint

        while not (term or trunc):
            action, _ = model.predict(
                obs,  # type: ignore[arg-type]
                deterministic=True,
            )
            action = action[0] if len(action) == 1 else action
            obs, rew, term, trunc, info = env.step(action)
            steps += 1

        # Log if the waypoint was reached or not
        if info.get("env_complete", False):
            reached.append(wp)
            reached_episode_lengths.append(steps)
        else:
            failed.append(wp)

    reached = np.array(reached)
    failed = np.array(failed)

    # Calculate success fraction and mean episode length
    success_fraction = len(reached) / num_eps
    mean_episode_length = np.mean(reached_episode_lengths)

    print(f"Number of Generated Waypoints: {num_eps}")
    print(f"Success Fraction: {success_fraction:.2f}")
    print(f"Mean Episode Length (for successful episodes): {mean_episode_length:.2f}")

    # Helper function to save and/or show plots
    def plot_and_save_waypoints(title, waypoints, color, name, file_name):
        fig = go.Figure()

        if waypoints.size > 0:
            fig.add_trace(go.Scatter3d(
                x=waypoints[:, 0],
                y=waypoints[:, 1],
                z=waypoints[:, 2],
                mode='markers',
                marker=dict(size=5, color=color, opacity=0.7),
                name=name
            ))

        fig.update_layout(
            title={"text": title, "x": 0.5},
            scene=dict(
                xaxis_title="X Axis",
                yaxis_title="Y Axis",
                zaxis_title="Z Axis",
            ),
        )

        if save_path is not None:
            file_path = os.path.join(save_path, file_name)
            fig.write_html(file_path + ".html")
            fig.write_image(f"{file_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
            print(f"Saved plot to {file_path}")

        fig.show()

    # Plot and save individual plots
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    plot_and_save_waypoints(f"Waypoints Reached ({np.round(success_fraction, 2)})", reached, 'green', 'Reached', "reached_waypoints")
    plot_and_save_waypoints(f"Waypoints Not Reached ({np.round(1 - success_fraction, 2)})", failed, 'red', 'Failed', "failed_waypoints")

    # Plot and save combined plot
    fig = go.Figure()

    if reached.size > 0:
        fig.add_trace(go.Scatter3d(
            x=reached[:, 0],
            y=reached[:, 1],
            z=reached[:, 2],
            mode='markers',
            marker=dict(size=5, color='green', opacity=0.7),
            name='Reached'
        ))

    if failed.size > 0:
        fig.add_trace(go.Scatter3d(
            x=failed[:, 0],
            y=failed[:, 1],
            z=failed[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', opacity=0.7),
            name='Failed'
        ))

    fig.update_layout(
        title={"text": f"Waypoints Reached and Not Reached (Total: {num_eps})", "x": 0.5},
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
        ),
    )

    if save_path is not None:
        combined_path = os.path.join(save_path, "combined_waypoints.html")
        fig.write_html(combined_path)
        fig.write_image(f"{combined_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        print(f"Saved plot to {combined_path}")

    fig.show()



def analyze_tb_logs(directory):
    # Initialize lists to store results from each log file
    best_episode_lengths = []
    best_frac_env_completes = []
    steps_at_best_frac_env = []
    normalized_rewards_per_timestep = []
    episode_lengths_at_best_frac_env = []
    agent_hz = 30.0

    # Loop through all log files in the directory
    for log_dir in glob.glob(os.path.join(directory, "*/")):
        try:
            # Load the TensorBoard logs using the EventAccumulator
            ea = event_accumulator.EventAccumulator(log_dir)
            ea.Reload()

            # Extract "eval/mean_ep_length", "eval/frac_env_complete", and "eval/mean_reward"
            if "eval/mean_ep_length" in ea.scalars.Keys() and "eval/frac_env_complete" in ea.scalars.Keys() and "eval/mean_reward" in ea.scalars.Keys():
                mean_ep_length_events = ea.Scalars("eval/mean_ep_length")
                frac_env_complete_events = ea.Scalars("eval/frac_env_complete")
                mean_reward_events = ea.Scalars("eval/mean_reward")

                # Find the max value for mean_ep_length
                best_mean_ep_length = max(event.value for event in mean_ep_length_events)
                best_episode_lengths.append(best_mean_ep_length / agent_hz)  # Convert to seconds

                # Find the max value and corresponding step for frac_env_complete
                best_frac_env_complete = max(frac_env_complete_events, key=lambda event: event.value)
                best_frac_env_completes.append(best_frac_env_complete.value)
                steps_at_best_frac_env.append(best_frac_env_complete.step)

                # Find the mean_reward and mean_ep_length at the step where frac_env_complete is the best
                step = best_frac_env_complete.step
                mean_ep_length_at_best = next(event.value for event in mean_ep_length_events if event.step == step)
                mean_reward_at_best = next(event.value for event in mean_reward_events if event.step == step)
                episode_lengths_at_best_frac_env.append(mean_ep_length_at_best / agent_hz)  # Convert to seconds

                # Calculate normalized reward per timestep
                if mean_ep_length_at_best > 0:
                    normalized_reward = mean_reward_at_best / mean_ep_length_at_best
                    normalized_rewards_per_timestep.append(normalized_reward)

        except Exception as e:
            print(f"Error processing log file in {log_dir}: {e}")

    # Calculate mean and standard deviation for all metrics
    mean_best_ep_length = np.mean(best_episode_lengths)
    std_best_ep_length = np.std(best_episode_lengths)

    mean_best_frac_env = np.mean(best_frac_env_completes)
    std_best_frac_env = np.std(best_frac_env_completes)

    mean_steps_at_best_frac = np.mean(steps_at_best_frac_env)
    std_steps_at_best_frac = np.std(steps_at_best_frac_env)

    mean_normalized_reward = np.mean(normalized_rewards_per_timestep)
    std_normalized_reward = np.std(normalized_rewards_per_timestep)

    mean_ep_length_at_best_frac = np.mean(episode_lengths_at_best_frac_env)
    std_ep_length_at_best_frac = np.std(episode_lengths_at_best_frac_env)

    # Print results
    print("-------------------------------")
    print(f"Results: {directory}")
    print(f"Average best episode length (seconds): {mean_best_ep_length:.2f} ± {std_best_ep_length:.2f}")
    print(f"Average best frac_env_complete: {mean_best_frac_env:.2f} ± {std_best_frac_env:.2f}")
    print(f"Average steps at best frac_env_complete: {mean_steps_at_best_frac:.2f} ± {std_steps_at_best_frac:.2f}")
    print(f"Average episode length at best frac_env_complete (seconds): {mean_ep_length_at_best_frac:.2f} ± {std_ep_length_at_best_frac:.2f}")
    print(f"Average normalized reward per timestep: {mean_normalized_reward:.4f} ± {std_normalized_reward:.4f}")
    print("-------------------------------")

'''
root = "../logs/tensorboard_log/Final/"
dirs = ["PPO/DefaultHyperConstantLearningRate", "PPO/TunedHyperConstantLearningRate",
        "PPO/TunedHyperLinearLearningRate", "PPO/TunedHyperExponentialLearningRate", "PPO/TunedHyperCosineLearningRate",
        "SAC/DefaultHyperConstantLearningRate", "SAC/TunedHyperConstantLearningRate", "SAC/TunedHyperLinearLearningRate",
        "SAC/TunedHyperExponentialLearningRate", "SAC/TunedHyperCosineLearningRate"]

for dir in dirs:
    analyze_tb_logs(os.path.join(root, dir))
'''

