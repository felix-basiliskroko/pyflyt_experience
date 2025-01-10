import numpy as np


class Reward:
    """
    Reward function for the quadrotor agent.
    The reward function is based on the following components:
    - Line of sight reward: Reward for heading towards the target
    - Smoothness reward: Reward for flying smooth trajectories

    :param r_LOS_weight: Weight determining the importance of the line of sight reward in the total reward signal.
    :param r_smooth_weight: Weight determining the importance of the smoothness reward in the total reward signal.
    :param steep_grad: Steepness of the reward function. Higher values lead to steeper reward gradients.
    :param flight_mode: Flight mode of the quadrotor agent. Currently only '1' is implemented.
    :param smooth_max: Maximum value of the smoothness reward. Used for normalization.
    :param reward_shift: Shifts the reward signal to the range [-1 + reward_shift, 0 + reward_shift]. 0 means only negative rewards and 1 means only positive rewards.
    """
    def __init__(self, r_LOS=None, r_smooth=None, smooth_max=None, steep_grad=1.0, flight_mode=1, reward_shift=0.0):
        self.r_LOS_weight = r_LOS
        self.r_smooth_weight = r_smooth
        self.steep_grad = steep_grad
        self.flight_mode = flight_mode
        self.smooth_max = smooth_max
        self.reward_shift = reward_shift
        print(f"Reward function initialized with r_LOS_weight: {self.r_LOS_weight}, r_smooth_weight: {self.r_smooth_weight}")
        print(f'Shifting reward to the range: [{-1 + reward_shift}, {0.0 + reward_shift}]')
        assert self.r_smooth_weight == 0.0, "Smooth reward is not implemented for reward shifting. Need to do that!"
        assert 0.0 <= self.reward_shift <= 1.0, f"Reward shift must be in the range [0.0, 1.0] but is {self.reward_shift}"

    def yield_reward(self, state, action):
        reward_signal = 0
        reward_components = {}

        # Line of sight reward for heading towards the target
        if self.r_LOS_weight is not None:
            r_LOS = -self.LOS_reward(state)
            r_LOS += self.reward_shift  # Bring reward to positive range
            weighted_r_LOS = self.r_LOS_weight * r_LOS
            reward_signal += weighted_r_LOS  # Add weighted LOS reward
            reward_components["los_reward"] = {"unweighted": r_LOS, "weighted": weighted_r_LOS}

        # Smoothness reward for flying smooth trajetories
        #TODO: Adapt to reward shift, atm not scaled to the proper range [-1, 0] but does not matter because r_smooth_weight is 0.0
        if self.r_smooth_weight is not None:
            r_smooth = -2 * (self.smooth_reward(state, action) / self.smooth_max)  # Normalize smooth reward to range -2 to 0
            r_smooth += self.reward_shift  # Bring reward to positive range
            weighted_r_smooth = self.r_smooth_weight * r_smooth
            reward_signal += weighted_r_smooth  # Add weighted smooth reward
            reward_components["smooth_reward"] = {"unweighted": r_smooth, "weighted": weighted_r_smooth}

        return reward_signal, reward_components

    def LOS_reward(self, state):  # Normalize to range [-1, 0]
        azimuth_reward = np.abs(0.5 * state["azimuth_angle"])**self.steep_grad
        elevation_reward = np.abs(0.5 * state["elevation_angle"])**self.steep_grad
        return azimuth_reward[0] + elevation_reward[0]

    def smooth_reward(self, state, action):
        if self.flight_mode == 1:
            return np.linalg.norm(state["ang_pos"] - np.delete(action, 3))  # Smooth control reward
        else:
            # raise NotImplementedError("Flight mode other than '1' not implemented")
            return 0.0
