import tensorflow as tf
from tensorflow import keras
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt


class ObservationHistCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ang_vels = []
        self.ang_poss = []
        self.lin_vels = []
        self.lin_poss = []
        self.quaternions = []
        self.prev_actions = []
        self.auxiliaries = []
        self.target_deltas = []
        self.roll_outs = 1

    def _on_step(self) -> bool:
        state = self.model.get_env().get_attr("state")[0]
        self.ang_vels += state["ang_vel"].tolist()
        self.ang_poss += state["ang_pos"].tolist()
        self.lin_vels += state["lin_vel"].tolist()
        self.lin_poss += state["lin_pos"].tolist()
        self.quaternions += state["quaternion"].tolist()
        self.prev_actions += state["prev_action"].tolist()
        self.auxiliaries += state["auxiliary"].tolist()
        self.target_deltas += state["target_delta"].tolist()

        return True

    def _on_rollout_start(self) -> None:
        self.ang_vels = []
        self.ang_poss = []
        self.lin_vels = []
        self.lin_poss = []
        self.quaternions = []
        self.prev_actions = []
        self.auxiliaries = []
        self.target_deltas = []
        return True

    def _on_rollout_end(self) -> None:
        fig_ang_vel = self.plot_distribution(self.ang_vels)
        self.logger.record("dist/ang_vel", Figure(fig_ang_vel, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_ang_pos = self.plot_distribution(self.ang_poss)
        self.logger.record("dist/ang_pos", Figure(fig_ang_pos, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_lin_vel = self.plot_distribution(self.lin_vels)
        self.logger.record("dist/lin_vel", Figure(fig_lin_vel, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_lin_pos = self.plot_distribution(self.lin_poss)
        self.logger.record("dist/lin_pos", Figure(fig_lin_pos, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_quaternion = self.plot_distribution(self.quaternions)
        self.logger.record("dist/quaternion", Figure(fig_quaternion, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_prev_action = self.plot_distribution(self.prev_actions)
        self.logger.record("dist/prev_action", Figure(fig_prev_action, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_auxiliary = self.plot_distribution(self.auxiliaries)
        self.logger.record("dist/auxiliary", Figure(fig_auxiliary, close=True), exclude=("stdout", "log", "json", "csv"))
        fig_target_delta = self.plot_distribution(self.target_deltas)
        self.logger.record("dist/target_delta", Figure(fig_target_delta, close=True), exclude=("stdout", "log", "json", "csv"))

        self.roll_outs += 1
        plt.close()
        return True

    def plot_distribution(self, obs):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Calculate histogram
        hist, bins = np.histogram(obs, bins=100)
        xs = (bins[:-1] + bins[1:]) / 2

        # Calculate the mean and std of observations
        mean = np.mean(obs)
        std = np.std(obs)

        # Plotting
        ax.bar(xs, hist, width=np.diff(bins), color='b', alpha=0.7)

        # Setting title with mean and std
        ax.set_title(f'Distribution with Mean: {mean:.2f}, Std Dev: {std:.2f} (n_rollout: {self.roll_outs})')
        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')

        return fig

from typing import List, Dict, Any
import numpy as np
import os


class StabilityEvalCallback(EvalCallback):
    def __init__(self, *args, threshold: float = 0.5, n_stability_epochs: int = 10, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.n_stability_epochs = n_stability_epochs
        self.recent_rewards: List[float] = []

    def _on_step(self) -> bool:
        continue_training = super()._on_step()

        # Only proceed if the recent rewards list is filled with enough evaluations
        if len(self.recent_rewards) >= self.n_stability_epochs:
            # Calculate the differences between consecutive evaluations
            differences = [abs(self.recent_rewards[i] - self.recent_rewards[i - 1]) for i in range(1, len(self.recent_rewards))]
            # Check if all differences are below the threshold
            if all(diff < self.threshold for diff in differences):
                print("Model performance is stable. Adjusting spawn point.")
                self.spawn_point_r_adjustment()

            # Remove the oldest reward to make room for new evaluations
            self.recent_rewards.pop(0)

        # Append the latest mean reward to the recent_rewards list
        self.recent_rewards.append(self.last_mean_reward)

        # Log spawn_point_radius
        self.logger.record("train/spawn_point_r", self.model.get_env().get_attr("spawn_point_r"))

        return continue_training

    def spawn_point_r_adjustment(self):
        """
        Adjust the spawn point attribute `self.spawn_point_r` based on your specific logic.
        """
        # Example adjustment logic
        old_spawn_point_r = self.model.get_env().get_attr("spawn_point_r")[0]
        self.model.get_env().set_attr(attr_name="spawn_point_r", value=old_spawn_point_r+0.05)
