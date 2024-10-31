from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class SingleWaypointQuadXEnv(QuadXBaseEnv):
    """QuadX Waypoints Environment.

    Actions are vp, vq, vr, T, ie: angular rates and thrust.
    The target is a set of `[x, y, z, (optional) yaw]` waypoints in space.

    Args:
    ----
        sparse_reward (bool): whether to use sparse rewards or not.
        num_targets (int): number of waypoints in the environment.
        use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoints for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
        flight_mode (int): the flight mode of the UAV.
        flight_dome_size (float): size of the allowable flying area.
        max_duration_seconds (float): maximum simulation time of the environment.
        angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
        agent_hz (int): looprate of the agent to environment interaction.
        render_mode (None | Literal["human", "rgb_array"]): render_mode
        render_resolution (tuple[int, int]): render_resolution.

    """

    def __init__(
            self,
            num_targets: int = 1,
            use_yaw_targets: bool = False,
            goal_reach_distance: float = 0.2,
            goal_reach_angle: float = 0.1,
            flight_mode: int = 0,
            flight_dome_size: float = 50.0,
            max_duration_seconds: float = 10.0,
            angle_representation: Literal["euler", "quaternion"] = "quaternion",
            agent_hz: int = 30,
            render_mode: None | Literal["human", "rgb_array"] = None,
            render_resolution: tuple[int, int] = (480, 480),
    ):
        """__init__.

        Args:
        ----
            sparse_reward (bool): whether to use sparse rewards or not.
            num_targets (int): number of waypoints in the environment.
            use_yaw_targets (bool): whether to match yaw targets before a waypoint is considered reached.
            goal_reach_distance (float): distance to the waypoints for it to be considered reached.
            goal_reach_angle (float): angle in radians to the waypoints for it to be considered reached, only in effect if `use_yaw_targets` is used.
            flight_mode (int): the flight mode of the UAV.
            flight_dome_size (float): size of the allowable flying area.
            max_duration_seconds (float): maximum simulation time of the environment.
            angle_representation (Literal["euler", "quaternion"]): can be "euler" or "quaternion".
            agent_hz (int): looprate of the agent to environment interaction.
            render_mode (None | Literal["human", "rgb_array"]): render_mode
            render_resolution (tuple[int, int]): render_resolution.

        """
        super().__init__(
            start_pos=np.array([[0.0, 0.0, 1.0]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.state = None
        self.reward_info = {
            "scaled_los_reward": 0.0,
            "scaled_altitude_reward": 0.0,
            "scaled_smooth_reward": 0.0,
        }

        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=0.1,
            np_random=self.np_random,
        )

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "t_azimuth_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "t_elevation_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "a_azimuth_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "a_elevation_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "aux_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
                "altitude": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
            })

    def reset(
            self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ):
        """Resets the environment.

        Args:
        ----
            seed: seed to pass to the base environment.
            options: None

        """
        super().begin_reset(seed, options)
        self.waypoints.reset(self.env, self.np_random)
        self.info["num_targets_reached"] = 0
        super().end_reset()

        self.state = {
            "t_azimuth_angle": np.zeros(1),
            "t_elevation_angle": np.zeros(1),
            "a_azimuth_angle": np.zeros(1),
            "a_elevation_angle": np.zeros(1),
            "aux_state": np.zeros(4),
            "altitude": np.zeros(1),
        }

        return self.state, self.info

    def compute_state(self) -> None:
        """Computes the state for a single waypoint environment."""
        # super().compute_state()
        # Since there's only one waypoint, we take the first and only target delta
        if self.angle_representation == 1:
            _, _, lin_vel, lin_pos, _ = super().compute_attitude()

            t_xy_proj, a_xy_proj = self.waypoints.targets[0][:2], lin_vel[:2]
            t_az_ang = np.arctan2(t_xy_proj[1], t_xy_proj[0])
            a_az_ang = np.arctan2(a_xy_proj[1], a_xy_proj[0])

            t_xz_proj, a_xz_proj = self.waypoints.targets[0][[0, 2]], lin_vel[[0, 2]]
            t_el_ang = np.arctan2(t_xz_proj[1], t_xz_proj[0])
            a_el_ang = np.arctan2(a_xz_proj[1], a_xz_proj[0])

            new_state = dict()
            new_state["t_azimuth_angle"] = np.array([t_az_ang])
            new_state["t_elevation_angle"] = np.array([t_el_ang])
            new_state["a_azimuth_angle"] = np.array([a_az_ang])
            new_state["a_elevation_angle"] = np.array([a_el_ang])
            new_state["aux_state"] = super().compute_auxiliary()
            new_state["altitude"] = np.array([lin_pos[2]]) if lin_pos[2] < 1.0 else np.array([1.0])
        else:
            raise NotImplementedError("Only quaternion representation is supported for now.")

        self.state = new_state

    def compute_term_trunc_reward(self):
        """Handle termination, truncation, and reward specifically for single waypoint."""
        if np.any(self.env.contact_array[self.env.planeId]):
            self.reward = -100.0
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.reward = -100.0
            self.info["out_of_bounds"] = True
            self.termination |= True

        # target reached
        if self.waypoints.target_reached:
            self.reward = 100.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached

        los_reward = ((np.pi/2 - np.abs(self.state["t_azimuth_angle"] - self.state["a_azimuth_angle"])) + (np.pi/2 - np.abs(self.state["t_elevation_angle"] - self.state["a_elevation_angle"]))) - np.pi
        # -4*pi if the agent is facing the opposite direction of the target; 0 if the agent is perfectly aligned with the target
        altitude_reward = self.state["altitude"]  # Negative altitude as reward
        smooth_reward = -np.linalg.norm(self.state["aux_state"] - self.action)  # Negative smooth control reward

        # Min-Max scaling to ensure all rewards are in the range [-2*pi, 0]
        scaled_los_reward = los_reward


        scaled_altitude_reward = (altitude_reward * 2*np.pi) - 2*np.pi

        smooth_reward_min = -np.linalg.norm(np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 0.8]))
        smooth_reward_max = 0.0
        scaled_smooth_reward = (((smooth_reward - smooth_reward_min) / (smooth_reward_max - smooth_reward_min)) * 2*np.pi) - 2*np.pi

        self.reward_info["scaled_los_reward"] = scaled_los_reward
        self.reward_info["scaled_altitude_reward"] = scaled_altitude_reward
        self.reward_info["scaled_smooth_reward"] = scaled_smooth_reward

        reward = 0.2 * scaled_los_reward + 0.6 * scaled_altitude_reward + 0.2 * scaled_smooth_reward
        self.reward = reward[0]
