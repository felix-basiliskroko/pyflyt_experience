import numpy as np

class Reward():
    def __init__(self, r_LOS=None, r_smooth=None, smooth_max=None, negative=True, steep_grad=1, flight_mode=1):
        self.r_LOS_weight = r_LOS
        self.r_smooth_weight = r_smooth
        self.steep_grad = steep_grad
        self.negative = negative
        self.flight_mode = flight_mode
        self.smooth_max = smooth_max
        print(f"Reward function initialized with r_LOS_weight: {self.r_LOS_weight}, r_smooth_weight: {self.r_smooth_weight}")
        print(f'Using negative reward: {self.negative}')

    def yield_reward(self, state, action):
        reward_signal = 0
        reward_components = {}

        # Line of sight reward for heading towards the target
        if self.r_LOS_weight is not None:
            r_LOS = -self.LOS_reward(state)
            if not self.negative:
                r_LOS += 2  # Bring reward to positive range
            weighted_r_LOS = self.r_LOS_weight * r_LOS
            reward_signal += weighted_r_LOS  # Add weighted LOS reward
            reward_components["los_reward"] = {"unweighted": r_LOS, "weighted": weighted_r_LOS}

        # Smoothness reward for flying smooth trajetories
        if self.r_smooth_weight is not None:
            r_smooth = -2 * (self.smooth_reward(state, action) / self.smooth_max)  # Normalize smooth reward to range -2 to 0
            if not self.negative:
                r_smooth += 2  # Bring reward to positive range
            weighted_r_smooth = self.r_smooth_weight * r_smooth
            reward_signal += weighted_r_smooth  # Add weighted smooth reward
            reward_components["smooth_reward"] = {"unweighted": r_smooth, "weighted": weighted_r_smooth}

        return reward_signal, reward_components

    def LOS_reward(self, state):
        azimuth_reward = np.abs(state["azimuth_angle"])**self.steep_grad
        elevation_reward = np.abs(state["elevation_angle"])**self.steep_grad
        return azimuth_reward[0] + elevation_reward[0]

    def smooth_reward(self, state, action):
        if self.flight_mode == 1:
            return np.linalg.norm(state["ang_pos"] - np.delete(action, 3))  # Smooth control reward
        else:
            raise NotImplementedError("Flight mode other than '1' not implemented")