import tensorflow as tf
from stable_baselines3.common.vec_env import sync_envs_normalization
from tensorflow import keras
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
from stable_baselines3.common.logger import Figure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
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


def _on_step(self) -> bool:
    continue_training = True

    if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                    "and warning above."
                ) from e

        # Reset success rate buffer
        self._is_success_buffer = []

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=self._log_success_callback,
        )

        if self.log_path is not None:
            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,
            )

        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        self.last_mean_reward = float(mean_reward)

        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
            print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Dump log so the evaluation results are printed with the correct timestep
        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path is not None:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = float(mean_reward)
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

    return continue_training