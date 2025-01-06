import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

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
            try:
                return float(value) if '.' in value else int(value)
            except ValueError:
                raise ValueError(f"Could not parse hyperparameter value from: {run_path}")
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
            sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))

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
            sorted_runs = sorted(dir_runs.items(), key=lambda x: x[0] if isinstance(x[0], (int, float)) else str(x[0]))
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
                        name=f'{run_label} ({os.path.basename(directory)})',
                        showlegend=len(all_dirs) > 1
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
