import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import re
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


'''
def extract_and_plot_tensorboard_logs(input_dirs, key_to_log, title, y_axis_label, average=False, save_path=None, log_scale=False, r=4):
    """
    Extract tensorboard information from one or multiple directories and plot the specified key.

    Parameters:
        input_dirs (str or list of str): Path(s) to the main directory/directories (e.g., 'steps=100000' or ['dir1', 'dir2']).
        key_to_log (str): The key to extract and log from tensorboard files.
        title (str): Title of the plot.
        y_axis_label (str): Label for the y-axis.
        average (bool): Whether to average the runs per directory (no std shading if multiple directories).
        save_path (str): Path to save the plot as a vector graphic (.pdf). If None, only display the plot.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    """
    pio.kaleido.scope.mathjax = None  # Disable MathJax rendering in Kaleido

    def extract_tensorboard_data(tfevent_file, key):
        """Extract data for a given key from a tensorboard file."""
        event_acc = EventAccumulator(tfevent_file)
        event_acc.Reload()
        if key not in event_acc.scalars.Keys():
            raise ValueError(f"Key '{key}' not found in {tfevent_file}.")
        events = event_acc.Scalars(key)
        steps, values = zip(*[(e.step, e.value) for e in events])
        return np.array(steps), np.array(values)

    def parse_run_value(run_path):
        """
        Extract the hyperparameter value from the run path.
        Handles paths like 'hyperparam_default_lr_mode_constant_run_1' or 'batch_size=64_1'.
        """
        if '=' in run_path:
            key, value = run_path.split('=')
            if '_' in value:
                value = value.split("_")[0]
            if re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', value):
                value = f"{float(value):.{len(str(value).split('.')[-1].rstrip('0'))}f}" if '.' in str(value) else str(value)
            elif value.isdigit():
                value = int(value)
            elif value.isalpha():
                value = str(value)
            else:
                raise ValueError(f"Could not parse hyperparameter value from: {run_path}")
            return value
        else:
            # Handle default naming without '='
            return "default"

    def adjust_for_log_scale(values):
        """Replace zero or negative values with a small positive constant for log scale."""
        return np.where(values <= 0, 1e-6, values)

    # Ensure input_dirs is a list
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    # Collect all TensorBoard log files
    all_dirs = {}
    for directory in input_dirs:
        dir_runs = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    tfevent_path = os.path.join(root, file)
                    run_name = os.path.basename(root)
                    try:
                        steps, values = extract_tensorboard_data(tfevent_path, key_to_log)
                        run_label = parse_run_value(run_name)
                        if run_label not in dir_runs:
                            dir_runs[run_label] = []
                        dir_runs[run_label].append((steps, values))
                    except ValueError as e:
                        print(f"Skipping file {tfevent_path}: {e}")
                        continue
        all_dirs[directory] = dir_runs

    # Initialize the figure
    fig = go.Figure()

    if average:
        multiple_dirs = len(all_dirs) > 1
        for directory, dir_runs in all_dirs.items():
            # sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            sorted_runs = sorted(dir_runs.items(), key=lambda x: (isinstance(x[0], (int, float)), x[0] if isinstance(x[0], (int, float)) else float('inf')))

            # Determine the minimum length of all runs
            min_length = min(
                [len(values) for _, runs in sorted_runs for _, values in runs]
            ) if sorted_runs else None

            if min_length is None:
                continue  # Skip directories with no valid runs

            # Compute mean values per directory
            all_steps = list(sorted_runs[0][1][0][0][:min_length])
            all_values = []
            for _, runs in sorted_runs:
                for steps, values in runs:
                    all_values.append(adjust_for_log_scale(values[:min_length]) if log_scale else values[:min_length])
            all_values = np.array(all_values)
            mean_values = np.mean(all_values, axis=0)

            # Add mean line for each directory
            fig.add_trace(go.Scatter(
                x=all_steps,
                y=mean_values,
                mode='lines',
                name='Average' if len(all_dirs) == 1 else f'Average ({os.path.basename(directory)})',
                line=dict()
            ))

            # Skip standard deviation shading if there are multiple directories
            if not multiple_dirs:
                std_values = np.std(all_values, axis=0)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([all_steps, all_steps[::-1]]),
                    y=np.concatenate([mean_values + std_values, (mean_values - std_values)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Standard deviation'
                ))
    else:
        for directory, dir_runs in all_dirs.items():
            # sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            sorted_runs = sorted(dir_runs.items(), key=lambda x: (isinstance(x[0], (int, float)), x[0] if isinstance(x[0], (int, float)) else float('inf')))
            min_length = min(
                [len(values) for _, runs in sorted_runs for _, values in runs]
            ) if sorted_runs else None

            if min_length is None:
                continue  # Skip directories with no valid runs

            for run_label, runs in sorted_runs:
                for steps, values in runs:
                    adjusted_values = adjust_for_log_scale(values[:min_length]) if log_scale else values[:min_length]
                    fig.add_trace(go.Scatter(
                        x=steps[:min_length],
                        y=adjusted_values,
                        mode='lines',
                        # name=f'{run_label} ({os.path.basename(directory)})',
                        name=run_label,
                        # showlegend=len(all_dirs) > 1
                        showlegend=True
                    ))

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Steps',
        yaxis_title=y_axis_label,
        yaxis=dict(
            type='log' if log_scale else 'linear',
            gridcolor='rgba(128,128,128,0.3)',
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.3)',
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        legend=dict(
            x=1.05,
            y=1,
            orientation='v'
        ) if len(all_dirs) > 1 else dict(),
        plot_bgcolor='rgba(255,255,255,0)',
        template="plotly_white"  # Forces light mode theme
    )

    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(f"{save_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        print(f"Figure saved at {save_path}.pdf")

    # Show the figure
    fig.show()
'''


def extract_and_plot_tensorboard_logs(input_dirs, key_to_log, title, y_axis_label, average=False, save_path=None, log_scale=False, r=4):
    """
    Extract tensorboard information from one or multiple directories and plot the specified key.

    Parameters:
        input_dirs (str or list of str): Path(s) to the main directory/directories (e.g., 'steps=100000' or ['dir1', 'dir2']).
        key_to_log (str): The key to extract and log from tensorboard files.
        title (str): Title of the plot.
        y_axis_label (str): Label for the y-axis.
        average (bool): Whether to average the runs per directory (no std shading if multiple directories).
        save_path (str): Path to save the plot as a vector graphic (.pdf). If None, only display the plot.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis.
    """
    pio.kaleido.scope.mathjax = None  # Disable MathJax rendering in Kaleido

    def extract_tensorboard_data(tfevent_file, key):
        """Extract data for a given key from a tensorboard file."""
        event_acc = EventAccumulator(tfevent_file)
        event_acc.Reload()
        if key not in event_acc.scalars.Keys():
            raise ValueError(f"Key '{key}' not found in {tfevent_file}.")
        events = event_acc.Scalars(key)
        steps, values = zip(*[(e.step, e.value) for e in events])  # Unpack into separate lists
        return np.array(steps), np.array(values)

    def parse_run_value(run_path):
        """
        Extract the hyperparameter value from the run path.
        Handles paths like 'hyperparam_default_lr_mode_constant_run_1' or 'batch_size=64_1'.
        """
        if '=' in run_path:
            key, value = run_path.split('=')
            if '_' in value:
                value = value.split("_")[0]
            if re.match(r'^-?\d+(\.\d+)?([eE][-+]?\d+)?$', value):
                value = f"{float(value):.{len(str(value).split('.')[-1].rstrip('0'))}f}" if '.' in str(value) else str(value)
            elif value.isdigit():
                value = int(value)
            elif value.isalpha():
                value = str(value)
            else:
                raise ValueError(f"Could not parse hyperparameter value from: {run_path}")
            return value
        else:
            # Handle default naming without '='
            return "default"

    def adjust_for_log_scale(values):
        """Replace zero or negative values with a small positive constant for log scale."""
        return np.where(values <= 0, 1e-6, values)

    # Ensure input_dirs is a list
    if isinstance(input_dirs, str):
        input_dirs = [input_dirs]

    # Collect all TensorBoard log files
    all_dirs = {}
    for directory in input_dirs:
        dir_runs = {}
        for root, _, files in os.walk(directory):
            for file in files:
                if file.startswith("events.out.tfevents"):
                    tfevent_path = os.path.join(root, file)
                    run_name = os.path.basename(root)
                    try:
                        steps, values = extract_tensorboard_data(tfevent_path, key_to_log)
                        run_label = parse_run_value(run_name)
                        if run_label not in dir_runs:
                            dir_runs[run_label] = []
                        dir_runs[run_label].append((steps, values))
                    except ValueError as e:
                        print(f"Skipping file {tfevent_path}: {e}")
                        continue
        all_dirs[directory] = dir_runs

    # Initialize the figure
    fig = go.Figure()

    if average:
        multiple_dirs = len(all_dirs) > 1
        for directory, dir_runs in all_dirs.items():
            # sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            sorted_runs = sorted(dir_runs.items(), key=lambda x: (isinstance(x[0], (int, float)), x[0] if isinstance(x[0], (int, float)) else float('inf')))

            # Create a common step grid
            all_steps = np.linspace(
                min([steps[0] for _, runs in sorted_runs for steps, _ in runs]),
                max([steps[-1] for _, runs in sorted_runs for steps, _ in runs]),
                num=1000  # Adjust number of points as needed
            )

            # Interpolate values for each run to align with all_steps
            all_values = []
            for _, runs in sorted_runs:
                for steps, values in runs:
                    interpolated_values = np.interp(all_steps, steps, values)
                    all_values.append(adjust_for_log_scale(interpolated_values) if log_scale else interpolated_values)

            all_values = np.array(all_values)
            mean_values = np.mean(all_values, axis=0)

            # Add mean line for each directory
            fig.add_trace(go.Scatter(
                x=all_steps,
                y=mean_values,
                mode='lines',
                name='Average' if len(all_dirs) == 1 else f'Average ({os.path.basename(directory)})',
                line=dict()
            ))

            # Skip standard deviation shading if there are multiple directories
            if not multiple_dirs:
                std_values = np.std(all_values, axis=0)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([all_steps, all_steps[::-1]]),
                    y=np.concatenate([mean_values + std_values, (mean_values - std_values)[::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,0,255,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Standard deviation'
                ))
    else:
        for directory, dir_runs in all_dirs.items():
            # sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
            sorted_runs = sorted(dir_runs.items(), key=lambda x: (isinstance(x[0], (int, float)), x[0] if isinstance(x[0], (int, float)) else float('inf')))

            # Create a common step grid
            all_steps = np.linspace(
                min([steps[0] for _, runs in sorted_runs for steps, _ in runs]),
                max([steps[-1] for _, runs in sorted_runs for steps, _ in runs]),
                num=1000  # Adjust number of points as needed
            )

            # Interpolate values for each run to align with all_steps
            all_values = []
            for _, runs in sorted_runs:
                for steps, values in runs:
                    interpolated_values = np.interp(all_steps, steps, values)
                    all_values.append(adjust_for_log_scale(interpolated_values) if log_scale else interpolated_values)

            all_values = np.array(all_values)
            mean_values = np.mean(all_values, axis=0)

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title='Steps',
        yaxis_title=y_axis_label,
        yaxis=dict(
            type='log' if log_scale else 'linear',
            gridcolor='rgba(128,128,128,0.3)',
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        xaxis=dict(
            gridcolor='rgba(128,128,128,0.3)',
            zerolinecolor='rgba(128,128,128,0.3)'
        ),
        legend=dict(
            x=1.05,
            y=1,
            orientation='v'
        ) if len(all_dirs) > 1 else dict(),
        plot_bgcolor='rgba(255,255,255,0)',
        template="plotly_white"  # Forces light mode theme
    )

    # Save figure if save_path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_image(f"{save_path}.pdf", height=800, width=1200, scale=2, engine="kaleido")
        print(f"Figure saved at {save_path}.pdf")

    # Show the figure
    fig.show()


import os
import re
import plotly.graph_objects as go
import tensorflow as tf
import pandas as pd


def extract_metric_from_tfevents(tfevent_file, key):
    """Extracts a metric from a TensorBoard event file."""
    event_acc = EventAccumulator(tfevent_file)
    event_acc.Reload()
    if key not in event_acc.scalars.Keys():
        raise ValueError(f"Key '{key}' not found in {tfevent_file}.")
    events = event_acc.Scalars(key)
    steps, values = zip(*[(e.step, e.value) for e in events])
    return np.array(steps), np.array(values)

def moving_average(values, window_size=30):
    """Applies a moving average filter to smooth the values while keeping the length consistent."""
    smoothed = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
    pad_size = len(values) - len(smoothed)
    return np.concatenate((np.full(pad_size, smoothed[0]), smoothed))  # Pad with the first value

def aggregate_runs(directory, metrics):
    """Aggregates multiple runs by averaging values and computing standard deviation."""
    aggregated_data = {}

    for run in sorted(os.listdir(directory)):
        run_path = os.path.join(directory, run)
        if not os.path.isdir(run_path):
            continue

        for file in os.listdir(run_path):
            if file.startswith("events.out.tfevents"):
                event_path = os.path.join(run_path, file)
                for metric in metrics:
                    try:
                        steps, values = extract_metric_from_tfevents(event_path, metric)
                        if metric not in aggregated_data:
                            aggregated_data[metric] = {}
                        for step, value in zip(steps, values):
                            if step not in aggregated_data[metric]:
                                aggregated_data[metric][step] = []
                            aggregated_data[metric][step].append(value)
                    except ValueError:
                        print(f"Metric '{metric}' not found in {event_path}, skipping.")
                    except Exception as e:
                        print(f"Error reading {event_path}: {e}")

    final_data = {}
    for metric in aggregated_data:
        final_data[metric] = {}
        for step in sorted(aggregated_data[metric].keys()):
            values = np.array(aggregated_data[metric][step])
            final_data[metric][step] = (np.mean(values), np.std(values))

    return final_data

def compare_training_configurations(name_var, directory, save_directory):
    """
    Visualizes training metrics from TensorBoard logs using Plotly, comparing different configurations.

    :param name_var: Name of the algorithm (e.g., "PPO")
    :param directory: Path to the directory containing subdirectories with different configurations
    :param save_directory: Path to the directory where the figures will be saved
    """

    import plotly.io as pio
    pio.kaleido.scope.mathjax = None  # Disable MathJax rendering in Kaleido
    metrics = {
        "eval/mean_reward": "Mean Reward over Evaluation Episodes",
        "rollout/ep_rew_mean": "Mean Reward over Training Episodes",
        "eval/frac_env_complete": "Percentage of Successful Evaluation Episodes"
    }

    comparisons = [
        ("DefaultHyperConstantLearningRate", "TunedHyperConstantLearningRate", "Angular Control (Default) vs. Angular Control (Tuned)"),
        ("TunedHyperConstantLearningRate", ["TunedHyperCosineLearningRate", "TunedHyperLinearLearningRate", "TunedHyperExponentialLearningRate"], "Angular Control (Constant LR) vs. Angular Control (Cosine, Linear, Exponential LR Decay)"),
        ("MThrustTunedHyperConstantLearningRate", "TunedHyperConstantLearningRate", "Thrust Control vs. Angular Control (Tuned)")
    ]

    legend_labels = {
        "DefaultHyperConstantLearningRate": "Angular Control with default hyperparameters",
        "TunedHyperConstantLearningRate": "Angular Control with tuned hyperparameters",
        "TunedHyperCosineLearningRate": "Angular Control (Tuned; Cosine LR Decay)",
        "TunedHyperLinearLearningRate": "Angular Control (Tuned; Linear LR Decay)",
        "TunedHyperExponentialLearningRate": "Angular Control (Tuned; Exponential LR Decay)",
        "MThrustTunedHyperConstantLearningRate": "Thrust Control with tuned hyperparameters"
    }

    os.makedirs(save_directory, exist_ok=True)

    for config_1, config_2, comparison_title in comparisons:
        data_1 = aggregate_runs(os.path.join(directory, config_1), metrics.keys())

        if isinstance(config_2, list):
            data_2 = {cfg: aggregate_runs(os.path.join(directory, cfg), metrics.keys()) for cfg in config_2}
        else:
            data_2 = {config_2: aggregate_runs(os.path.join(directory, config_2), metrics.keys())}

        for metric, y_label in metrics.items():
            fig = go.Figure()

            colors = ['blue', 'red', 'green', 'purple']

            for idx, (cfg, data) in enumerate([(config_1, data_1)] + list(data_2.items())):
                if metric in data:
                    steps, values = zip(*sorted(data[metric].items()))
                    mean_values, std_dev = zip(*values)

                    if metric == "rollout/ep_rew_mean":
                        mean_values = moving_average(np.array(mean_values))
                        std_dev = moving_average(np.array(std_dev))

                    fig.add_trace(go.Scatter(
                        x=steps, y=mean_values, mode='lines', name=legend_labels[cfg],
                        line=dict(width=2, color=colors[idx])
                    ))
                    if "LR Decay" not in comparison_title:
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([steps, steps[::-1]]),
                            y=np.concatenate([np.array(mean_values) - np.array(std_dev),
                                              (np.array(mean_values) + np.array(std_dev))[::-1]]),
                            fill='toself', fillcolor=f'rgba({0 if colors[idx] == "blue" else 255},0,{255 if colors[idx] == "blue" else 0},0.2)',
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False
                        ))

            if not fig.data:
                print(f"No data available for {metric}, skipping plot.")
                continue

            fig.update_layout(
                xaxis=dict(
                    gridcolor='rgba(0,0,0,0.08)',  # Sets gridlines to black with 0.08 opacity
                    tickmode='linear',
                    dtick=100000,
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                yaxis=dict(
                    gridcolor='rgba(0,0,0,0.08)',  # Same adjustment for y-axis
                    title_font=dict(size=14),
                    tickfont=dict(size=12)
                ),
                template="seaborn",
                title=dict(text=f"{name_var}: {comparison_title}", x=0.5, font=dict(size=18, family="Arial", color="black")),
                xaxis_title="Number of Steps",
                yaxis_title=y_label,
                showlegend=True,
                plot_bgcolor='white',
                paper_bgcolor='white',
                width=1200
            )

            metric_save_path = os.path.join(save_directory, f"{name_var}_{comparison_title.replace(' ', '_')}_{metric.replace('/', '_')}.pdf")
            pio.write_image(fig, metric_save_path, format='pdf')
            print(f"Plot saved to {metric_save_path}")


'''
dir = '../train/hyperparameters/logs/hyper_sac/lr'
name_var = "SAC"
param_name = "Learning Rate"
save_path = "hyperparams/sac/hyperparams/lr_tuning.pdf"
visualize_training_metrics(name_var=name_var, param_name=param_name, save_path=save_path, directory=dir)

dir = '../train/hyperparameters/logs/hyper_ppo/gamma'
name_var = "PPO"
param_name = "Gamma"
save_path = "hyperparams/ppo/hyperparams/gamma_tuning.pdf"
visualize_training_metrics(name_var=name_var, param_name=param_name, save_path=save_path, directory=dir)

dir = '../train/hyperparameters/logs/hyper_ddpg/lr'
name_var = "DDPG"
param_name = "Learning Rate"
save_path = "hyperparams/ddpg/hyperparams/lr_tuning.pdf"
visualize_training_metrics(name_var=name_var, param_name=param_name, save_path=save_path, directory=dir)
'''
dir = "../logs/tensorboard_log/Final/SAC"
name = "SAC"
save_dir = "angular_control/hyperparams/sac/final_runs"

compare_training_configurations(name_var=name, directory=dir, save_directory=save_dir)
