import numpy as np

class Normaliser:
    def __init__(self, alpha: float):
        # Initial parameters approximated from 100_000 random steps in the environment (see ema_parameters.json for reference)
        self.alpha = alpha
        self.lin_pos_mu_x, self.lin_pos_mu_y, self.lin_pos_mu_z = -39.6995, -34.2892, 41.4953
        self.lin_pos_sigma_x, self.lin_pos_sigma_y, self.lin_pos_sigma_z = 34.0776, 31.3624, 23.4024

        self.lin_vel_mu_x, self.lin_vel_mu_y, self.lin_vel_mu_z = -6.9273, -3.1629, 14.1149
        self.lin_vel_sigma_x, self.lin_vel_sigma_y, self.lin_vel_sigma_z = 5.2731, 5.6165, 4.9053

        self.prev_dist_mu = 2075.6744
        self.prev_dist_sigma = 11.6396

        self.ang_vel_min, self.ang_vel_max = -np.pi, np.pi
        self.ang_pos_min, self.ang_pos_max = -np.pi, np.pi
        self.quat_min, self.quat_max = -1.0, 1.0

        self.thrust_min, self.thrust_max = -1.0, 1.0

    def _ema_parameter_updates(self, attitude) -> None:
        """
        Update mean and standard deviation of the states using the Exponential Moving Average (EMA).
        :param states:
        :return:
        """
        # Linear velocity
        self.lin_vel_mu_x = self.alpha * attitude[6] + (1 - self.alpha) * self.lin_vel_mu_x
        self.lin_vel_mu_y = self.alpha * attitude[7] + (1 - self.alpha) * self.lin_vel_mu_y
        self.lin_vel_mu_z = self.alpha * attitude[8] + (1 - self.alpha) * self.lin_vel_mu_z

        squared_diff_lin_vel_x = (attitude[6] - self.lin_vel_mu_x) ** 2
        self.lin_vel_sigma_x = np.sqrt(self.alpha * squared_diff_lin_vel_x + (1 - self.alpha) * self.lin_vel_sigma_x)

        squared_diff_lin_vel_y = (attitude[7] - self.lin_vel_mu_y) ** 2
        self.lin_vel_sigma_y = np.sqrt(self.alpha * squared_diff_lin_vel_y + (1 - self.alpha) * self.lin_vel_sigma_y)

        squared_diff_lin_vel_z = (attitude[8] - self.lin_vel_mu_z) ** 2
        self.lin_vel_sigma_z = np.sqrt(self.alpha * squared_diff_lin_vel_z + (1 - self.alpha) * self.lin_vel_sigma_z)

        # Linear position
        self.lin_pos_mu_x = self.alpha * attitude[9] + (1 - self.alpha) * self.lin_pos_mu_x
        self.lin_pos_mu_y = self.alpha * attitude[10] + (1 - self.alpha) * self.lin_pos_mu_y
        self.lin_pos_mu_z = self.alpha * attitude[11] + (1 - self.alpha) * self.lin_pos_mu_z

        squared_diff_lin_pos_x = (attitude[9] - self.lin_pos_mu_x) ** 2
        self.lin_pos_sigma_x = np.sqrt(self.alpha * squared_diff_lin_pos_x + (1 - self.alpha) * self.lin_pos_sigma_x)

        squared_diff_lin_pos_y = (attitude[10] - self.lin_pos_mu_y) ** 2
        self.lin_pos_sigma_y = np.sqrt(self.alpha * squared_diff_lin_pos_y + (1 - self.alpha) * self.lin_pos_sigma_y)

        squared_diff_lin_pos_z = (attitude[11] - self.lin_pos_mu_z) ** 2
        self.lin_pos_sigma_z = np.sqrt(self.alpha * squared_diff_lin_pos_z + (1 - self.alpha) * self.lin_pos_sigma_z)

    def ema_normaliser(self, attitude, aux_state, prev_action, target_delta, prev_distance) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Normalise the states using the Exponential Moving Average (EMA) normalisation technique.
        This excludes angular_velocity, angular_position and quaternion, since their min and max values are already known,
         hence min-max normalization can be applied to the range of .

        Args:
        ----
            states (list): list of states to be normalised.

        Returns:
        -------
            norm_states: list of normalised states.
        """
        # Update mean and standard deviation
        self._ema_parameter_updates(attitude=attitude)

        # Attitude
        # Angular velocity:
        norm_ang_vel = attitude[0:3]

        # Angular position:
        norm_ang_pos = attitude[3:6]

        # Linear velocity
        norm_lin_vel_x = np.array([(attitude[6] - self.lin_vel_mu_x) / self.lin_vel_sigma_x])
        norm_lin_vel_y = np.array([(attitude[7] - self.lin_vel_mu_y) / self.lin_vel_sigma_y])
        norm_lin_vel_z = np.array([(attitude[8] - self.lin_vel_mu_z) / self.lin_vel_sigma_z])

        # Linear position
        norm_lin_pos_x = np.array([(attitude[9] - self.lin_pos_mu_x) / self.lin_pos_sigma_x])
        norm_lin_pos_y = np.array([(attitude[10] - self.lin_pos_mu_y) / self.lin_pos_sigma_y])
        norm_lin_pos_z = np.array([(attitude[11] - self.lin_pos_mu_z) / self.lin_pos_sigma_z])

        # Quaternion
        norm_quat = attitude[11:15]

        norm_attitude = np.concatenate([norm_ang_vel, norm_ang_pos, norm_lin_vel_x, norm_lin_vel_y, norm_lin_vel_z, norm_lin_pos_x, norm_lin_pos_y, norm_lin_pos_z, norm_quat])

        # Auxiliary (Min-Max Normalisation from 0-1 to -pi, pi)
        norm_aux_1 = aux_state[0] * (2 * np.pi) - np.pi
        norm_aux_2 = aux_state[1] * (2 * np.pi) - np.pi
        norm_aux_3 = aux_state[2] * (2 * np.pi) - np.pi
        norm_aux_4 = aux_state[3] * (2 * np.pi) - np.pi
        norm_aux = np.array([norm_aux_1, norm_aux_2, norm_aux_3, norm_aux_4])

        # Previous action (Min-Max Normalisation from 0-1, to -pi, pi)
        norm_prev_action = prev_action * (2 * np.pi) - np.pi

        # Target delta
        norm_target_delta = target_delta / np.linalg.norm(target_delta)

        # Previous distance
        norm_prev_dist = prev_distance

        return norm_attitude, norm_aux, norm_prev_action, norm_target_delta, norm_prev_dist

    def return_paramters(self):
        print(f'Linear Velocity (Mean of x): {self.lin_vel_mu_x}\n Linear Velocity (Mean of y): {self.lin_vel_mu_y}\n Linear Velocity (Mean of z): {self.lin_vel_mu_z}')
        print(f'Linear Velocity (Sigma of x): {self.lin_vel_sigma_x}\n Linear Velocity (Sigma of y): {self.lin_vel_sigma_y}\n Linear Velocity (Sigma of z): {self.lin_vel_sigma_z}')
        print(f'Linear Position (Mean of x): {self.lin_pos_mu_x}\n Linear Position (Mean of y): {self.lin_pos_mu_y}\n Linear Position (Mean of z): {self.lin_pos_mu_z}')
        print(f'Linear Position (Sigma of x): {self.lin_pos_sigma_x}\n Linear Position (Sigma of y): {self.lin_pos_sigma_y}\n Linear Position (Sigma of z): {self.lin_pos_sigma_z}')
