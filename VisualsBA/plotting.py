import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_and_plot_tensorboard_logs(directory, key_to_log, average=False):
    """
    Extract tensorboard information from subdirectories and plot the specified key.

    Parameters:
        directory (str): Path to the main directory (e.g., 'steps=100000').
        key_to_log (str): The key to extract and log from tensorboard files.
        average (bool): Whether to average the runs and display std shading.
    """
    def extract_tensorboard_data(tfevent_file, key):
        """Extract data for a given key from a tensorboard file."""
        event_acc = EventAccumulator(tfevent_file)
        event_acc.Reload()
        if key not in event_acc.scalars.Keys():
            raise ValueError(f"Key '{key}' not found in {tfevent_file}.")
        events = event_acc.Scalars(key)
        steps, values = zip(*[(e.step, e.value) for e in events])
        return np.array(steps), np.array(values)

    # Collect all runs
    all_runs = {}
    for root, _, files in os.walk(directory):
        for file in files:
            if file.startswith("events.out.tfevents"):
                tfevent_path = os.path.join(root, file)
                run_name = os.path.basename(root)
                try:
                    steps, values = extract_tensorboard_data(tfevent_path, key_to_log)
                    all_runs[run_name] = (steps, values)
                except ValueError as e:
                    print(e)
                    continue

    # Plot results
    fig = go.Figure()
    if average:
        # Average all runs
        all_steps = list(all_runs.values())[0][0]  # Assuming steps are identical for all runs
        all_values = np.array([values for _, values in all_runs.values()])
        mean_values = np.mean(all_values, axis=0)
        std_values = np.std(all_values, axis=0)

        # Add average line
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
        # Plot all individual runs
        for run_name, (steps, values) in all_runs.items():
            fig.add_trace(go.Scatter(
                x=steps,
                y=values,
                mode='lines',
                name=run_name
            ))

    # Layout settings
    fig.update_layout(
        title=f"Tensorboard Logs: '{key_to_log}'",
        xaxis_title='Steps',
        yaxis_title=key_to_log,
        template="plotly",
        legend=dict(x=0, y=1)
    )
    fig.show()
