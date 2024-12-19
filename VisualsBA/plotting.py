import os
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_and_plot_tensorboard_logs(directory, key_to_log, title, y_axis_label, average=False, save_path=None, log_scale=False, r=4):
    """
    Extract tensorboard information from subdirectories and plot the specified key.

    Parameters:
        directory (str): Path to the main directory (e.g., 'steps=100000').
        key_to_log (str): The key to extract and log from tensorboard files.
        title (str): Title of the plot.
        y_axis_label (str): Label for the y-axis.
        average (bool): Whether to average the runs and display std shading.
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
        Extract the hyperparameter value (e.g., gae=0.9 or ent_coeff=0.01) from the run path.
        Ignores intermediate directories like 'PPO_1'.
        """
        key, value = run_path.split('=')
        if '_' in value:
            value = value.split("_")[0]
        try:
            return float(value) if '.' in value else int(value)
        except ValueError:
            raise ValueError(f"Could not parse hyperparameter value from: {run_path}")

    def adjust_for_log_scale(values):
        """Replace zero or negative values with a small positive constant for log scale."""
        return np.where(values <= 0, 1e-6, values)

    # Collect all TensorBoard log files
    all_runs = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("events.out.tfevents"):
                tfevent_path = os.path.join(root, file)
                run_name = os.path.basename(root)
                try:
                    steps, values = extract_tensorboard_data(tfevent_path, key_to_log)
                    batch_size = parse_run_value(run_name)
                    all_runs[batch_size] = (steps, values)
                except ValueError as e:
                    print(f"Skipping file {tfevent_path}: {e}")
                    continue

    # Sort runs by batch size or hyperparameter
    sorted_runs = sorted(all_runs.items(), key=lambda x: x[0])

    # Compute y-axis limits for log scale
    all_y_values = []
    for _, (_, values) in sorted_runs:
        adjusted_values = adjust_for_log_scale(values) if log_scale else values
        all_y_values.extend(adjusted_values)
    y_min = min(v for v in all_y_values if v > 0)  # Smallest positive value
    y_max = max(all_y_values)

    # Initialize the figure
    fig = go.Figure()

    if average:
        # Compute mean and standard deviation
        all_steps = list(sorted_runs[0][1][0])
        all_values = np.array([adjust_for_log_scale(values) if log_scale else values for _, (_, values) in sorted_runs])
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        # Add mean line
        fig.add_trace(go.Scatter(
            x=all_steps,
            y=mean_values,
            mode='lines',
            name='Average',
            line=dict(color='blue')
        ))
        # Add standard deviation shading
        fig.add_trace(go.Scatter(
            x=np.concatenate([all_steps, all_steps[::-1]]),
            y=np.concatenate([mean_values + std_values, (mean_values - std_values)[::-1]]),
            fill='toself',
            fillcolor='rgba(0,0,255,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Standard Deviation'
        ))
    else:
        # Plot individual runs
        for batch_size, (steps, values) in sorted_runs:
            adjusted_values = adjust_for_log_scale(values) if log_scale else values
            if r:
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=adjusted_values,
                    mode='lines',
                    name=str(round(batch_size, r))
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=steps,
                    y=adjusted_values,
                    mode='lines',
                    name=str(batch_size)
                ))

    # Update layout
    if title:
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
                range=[np.log10(y_min), np.log10(y_max)] if log_scale else None,
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
            ),
            plot_bgcolor='rgba(255,255,255,0)',
            template="plotly_white"  # Forces light mode theme
        )
    else:
        fig.update_layout(
            xaxis_title='Steps',
            yaxis_title=y_axis_label,
            yaxis=dict(
                type='log' if log_scale else 'linear',
                range=[np.log10(y_min), np.log10(y_max)] if log_scale else None,
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
            ),
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
