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
            flight_dome_size: float = 6.0,
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
        self.orn_height = 3.0

        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=self.orn_height,
            np_random=self.np_random,
        )

        # init_LOS = self.waypoints.targets[0] - np.array([[0.0, 0.0, self.orn_height]])
        # unit_init_LOS = init_LOS/np.linalg.norm(init_LOS)

        super().__init__(
            start_pos=np.array([[0.0, 0.0, self.orn_height]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # define waypoints
        self.state = None
        self.xyz_limit = np.pi
        self.thrust_limit = 0.8

        # Reward scaling to be in the range [-2*pi, 0]
        self.smooth_max = np.linalg.norm(np.array([2*self.xyz_limit, 2*self.xyz_limit, 2*self.xyz_limit, self.thrust_limit]))

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "azimuth_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "elevation_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "aux_state": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "altitude": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
            })

        high = np.array(
            [
                self.xyz_limit,
                self.xyz_limit,
                self.xyz_limit,
                self.thrust_limit,
            ]
        )
        low = np.array(
            [
                -self.xyz_limit,
                -self.xyz_limit,
                -self.xyz_limit,
                0.0,
            ]
        )
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float64)

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
            "azimuth_angle": np.zeros(1),
            "elevation_angle": np.zeros(1),
            "aux_state": np.zeros(4),
            "ang_vel": np.zeros(3),
            "altitude": np.zeros(1),
        }

        self.info["waypoint"] = self.waypoints.targets[0]

        return self.state, self.info

    def ang(self, v1, v2):
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def compute_state(self) -> None:
        """Computes the state for a single waypoint environment."""

        if self.angle_representation == 1:
            ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
            LOS = self.waypoints.targets[0] - lin_pos
            LOS_xy_proj, LOS_xz_proj = LOS[:2]/np.linalg.norm(LOS[:2]), LOS[[0, 2]]/np.linalg.norm(LOS[[0, 2]])
            vel_xy_proj, vel_xz_proj = lin_vel[:2]/np.linalg.norm(lin_vel[:2]), lin_vel[[0, 2]]/np.linalg.norm(lin_vel[[0, 2]])

            az_ang = self.ang(LOS_xy_proj, vel_xy_proj)
            el_ang = self.ang(LOS_xz_proj, vel_xz_proj)

            assert np.all(np.abs([az_ang, el_ang]) <= np.pi), "Angles should be in the range [-pi, pi]"

            new_state = dict()
            new_state["azimuth_angle"] = np.array([az_ang])
            new_state["elevation_angle"] = np.array([el_ang])
            new_state["aux_state"] = super().compute_auxiliary()
            new_state["ang_vel"] = ang_vel
            new_state["altitude"] = np.array([lin_pos[2]]) if lin_pos[2] < self.orn_height else np.array([self.orn_height])

            # Store non-observable states (for debugging/evaluation purposes)
            self.info["angular_position"] = ang_pos
            self.info["quaternion"] = quaternion
            self.info["linear_position"] = lin_pos
            self.info["linear_velocity"] = lin_vel
            self.info["distance_to_target"] = np.linalg.norm(LOS)

        else:
            raise NotImplementedError("Only quaternion representation is supported for now.")

        self.state = new_state

    def compute_term_trunc_reward(self):
        # los_reward = np.abs(self.state["t_azimuth_angle"] - self.state["a_azimuth_angle"]) + np.abs(self.state["t_elevation_angle"] - self.state["a_elevation_angle"])
        # Each term (azimuth and elevation): [0, pi] -> [0, 2*pi] (where 0 means perfect alignment and pi means 180 degree misalignment)

        # los_reward = np.pi - np.abs(np.abs(self.state["azimuth_angle"]) - np.pi) + np.pi - np.abs(np.abs(self.state["elevation_angle"]) - np.pi)
        los_reward = np.abs(self.state["azimuth_angle"]) + np.abs(self.state["elevation_angle"])
        # Each term (azimuth and elevation): [0, pi] -> [0, 2*pi] (where 0 means perfect alignment and pi means 180 degree misalignment)

        smooth_reward = np.linalg.norm(self.state["aux_state"] - self.action)  # Smooth control reward

        # Min-Max scaling to ensure all rewards are in the range [-2*pi, 0]
        scaled_los_reward = -(los_reward)
        scaled_smooth_reward = (smooth_reward/self.smooth_max)*(-2*np.pi)

        self.info["reward_components"] = {
            "scaled_los_reward": scaled_los_reward,
            "scaled_smooth_reward": scaled_smooth_reward,
        }

        assert -2*np.pi <= scaled_los_reward <= 0, f"LOS reward should be in the range [-2*pi, 0] but got {scaled_los_reward}"
        # assert -2*np.pi <= scaled_smooth_reward <= 0, f"Smooth reward should be in the range [-2*pi, 0] but got {scaled_smooth_reward}"

        reward = 1.0 * scaled_los_reward + 0.0 * scaled_smooth_reward
        self.reward = reward[0]

        """Handle termination, truncation, and reward specifically for single waypoint."""
        if np.any(self.env.contact_array[self.env.planeId]):
            self.reward = -500.0
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            # self.reward = -500.0
            self.info["out_of_bounds"] = True
            self.termination |= True

        # target reached
        if self.waypoints.target_reached:
            self.reward = 500.0

            # advance the targets
            self.waypoints.advance_targets()

            # update infos and dones
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached

    def get_info(self):
        return self.info
