from __future__ import annotations

from typing import Any, Literal

import numpy as np
from PyFlyt.core import Aviary
from gymnasium import spaces
import pybullet as p

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler

from Envs.Rewards.reward import Reward


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
            goal_reach_distance: float = 1.0,
            goal_reach_angle: float = 0.1,
            flight_mode: int = 1,
            flight_dome_size: float = 30.0,
            max_duration_seconds: float = 10.0,
            angle_representation: Literal["euler", "quaternion"] = "quaternion",
            agent_hz: int = 30,
            render_mode: None | Literal["human", "rgb_array"] = None,
            render_resolution: tuple[int, int] = (480, 480),
            steep_grad: float = 1.0,
            reward_shift: float = 0.0,
            min_height: float = 0.6,
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

        self.start_height = 0.1*flight_dome_size  # start in the middle of the flight dome

        super().__init__(
            start_pos=np.array([[0.0, 0.0, self.start_height]]),
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            start_orn=np.array([[0.0, 0.0, 0.0]]),
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        self.waypoints = WaypointHandler(
            enable_render=self.render_mode is not None,
            num_targets=num_targets,
            use_yaw_targets=use_yaw_targets,
            goal_reach_distance=goal_reach_distance,
            goal_reach_angle=goal_reach_angle,
            flight_dome_size=flight_dome_size,
            min_height=min_height*flight_dome_size,
            np_random=self.np_random,
        )

        self.goal_reach_distance = goal_reach_distance
        self.state = None
        self.xyz_limit = np.pi
        self.thrust_limit = 0.8

        # Define observation space
        self.observation_space = spaces.Dict(
            {
                "azimuth_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "elevation_angle": spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float64),
                "ang_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                "altitude": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64),
            })

        # For Reward scaling
        smooth_max = np.linalg.norm(np.array([2*self.xyz_limit, 2*self.xyz_limit, 2*self.xyz_limit]))

        self.reached_reward = 100.0
        self.crash_reward = -100.0
        self.unstable_reward = -100.0
        self.reward_func = Reward(r_LOS=1.0, r_smooth=0.0, smooth_max=smooth_max,
                                  flight_mode=flight_mode,
                                  steep_grad=steep_grad,
                                  reward_shift=reward_shift)

        self.flight_mode = flight_mode
        # -1: m1, m2, m3, m4 (motor thrusts)
        # 0: vp, vq, vr, T (angular velocities and thrust)
        # 1: p, q, r, vz (angular positions and vertical velocity)
        # 2: vp, vq, vr, z (angular velocities and altitude)
        # 3: p, q, r, z (angular positions and altitude)
        # 4: u, v, vr, z (local linear velocities and altitude)
        # 5: u, v, vr, vz (local linear velocities, angular velocities, and vertical velocity)
        # 6: vx, vy, vr, vz (global linear velocities and angular velocities)
        # 7: x, y, r, z (global linear positions)

        if self.flight_mode == 1:  # p, q, r, vZ
            self.action_space = spaces.Box(
                low=np.array([-self.xyz_limit, -self.xyz_limit, -self.xyz_limit, -1.0]),
                high=np.array([self.xyz_limit, self.xyz_limit, self.xyz_limit, 1.0]),
                dtype=np.float64,
            )

        if self.flight_mode == 7:  # x, y, r, z
            self.action_space = spaces.Box(
                low=np.array([-np.inf, -np.inf, -self.xyz_limit, 0.0]),
                high=np.array([np.inf, np.inf, self.xyz_limit, np.inf]),
                dtype=np.float64,
            )

        elif self.flight_mode == "nudge":
            nudge = 0.1
            self.action_space = spaces.Box(
                low=np.array([-nudge*self.xyz_limit, -nudge*self.xyz_limit, -nudge*self.xyz_limit, -nudge*self.thrust_limit]),
                high=np.array([nudge*self.xyz_limit, nudge*self.xyz_limit, nudge*self.xyz_limit, nudge*self.thrust_limit]),
                dtype=np.float64,
            )

    def step(self, action: np.ndarray):
        if self.flight_mode == "nudge":  # Control via small nudges in pitch, yaw, roll, thrust
            pitch, yaw, roll, thrust = super().compute_auxiliary()
            new_pitch = np.clip(pitch + action[0], -self.xyz_limit, self.xyz_limit)
            new_yaw = np.clip(yaw + action[1], -self.xyz_limit, self.xyz_limit)
            new_roll = np.clip(roll + action[2], -self.xyz_limit, self.xyz_limit)
            new_thrust = np.clip(thrust + action[3], 0, self.thrust_limit)

            action = np.array([new_pitch, new_yaw, new_roll, new_thrust])
        return super().step(action)

    def reset(
            self, *, seed: None | int = None, options: None | dict[str, Any] = dict()
    ):
        """Resets the environment.

        Args:
        ----
            seed: seed to pass to the base environment.
            options: None

        """

        super().begin_reset(seed, options)  # self.env with neutral starting orientation
        self.waypoints.reset(self.env, self.np_random)  # Initialize the target vector

        # Calculate azimuth-aligned starting orientation based on target vector
        target_proj = self.waypoints.targets[0][:2]/np.linalg.norm(self.waypoints.targets[0][:2])
        init_yaw = self.ang(np.array([0.0, -1.0]), target_proj)
        self.start_orn = np.array([[0.0, 0.0, init_yaw]])

        # Overwrite self.env with updated starting orientation (self.start_orn)
        drone_options = dict()
        drone_options["use_camera"] = drone_options.get("use_camera", False) or bool(
            self.render_mode
        )
        drone_options["camera_fps"] = int(120 / self.env_step_ratio)

        self.env = Aviary(
            start_pos=self.start_pos,
            start_orn=self.start_orn,
            drone_type="quadx",
            render=self.render_mode == "human",
            drone_options=drone_options,
            seed=seed,
        )

        if self.render_mode == "human":
            self.camera_parameters = self.env.getDebugVisualizerCamera()

        # Explanation for above: begin_reset() needs to be called before waypoint/target can be set. Since begin_reset()
        # sets the self.env variable with the the orientation to neutral, self.env needs to be overwritten with the
        # updated starting orientation. The orientation is set to be azimuth-aligned with the target vector.
        #TODO Messy workaround. Find a better way to handle this. Maybe overwrite self.begin_reset()?

        self.action = np.zeros(4)
        self.info["num_targets_reached"] = 0
        super().end_reset()

        self.state = {
            "azimuth_angle": np.zeros(1),
            "elevation_angle": np.zeros(1),
            "ang_pos": np.zeros(3),
            "ang_vel": np.zeros(3),
            "altitude": np.zeros(1),
        }

        self.info["waypoint"] = self.waypoints.targets[0]
        self.info["unstable"] = False
        self.prev_pos = np.array([0.0, 0.0, self.start_height])

        return self.state, self.info

    def ang(self, v1, v2):
        #TODO move this to utils (static method)
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def compute_state(self) -> None:
        """Computes the state for a single waypoint environment."""

        if self.angle_representation == 1:
            ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()

            # Convert from body frame to global frame (https://taijunjet.com/PyFlyt/documentation/core/aviary.html#PyFlyt.core.Aviary.all_states)
            rotation = np.array(p.getMatrixFromQuaternion(quaternion)).reshape(3, 3)
            lin_vel = np.matmul(lin_vel, rotation.T)

            LOS = self.waypoints.targets[0] - lin_pos
            LOS_xy_proj, LOS_xz_proj = LOS[:2]/(np.linalg.norm(LOS[:2]) + 1e-10), LOS[[0, 2]]/(np.linalg.norm(LOS[[0, 2]]) + 1e-10)
            vel_xy_proj, vel_xz_proj = lin_vel[:2]/(np.linalg.norm(lin_vel[:2]) + 1e-10), lin_vel[[0, 2]]/(np.linalg.norm(lin_vel[[0, 2]]) + 1e-10)

            az_ang = self.ang(LOS_xy_proj, vel_xy_proj)
            el_ang = self.ang(LOS_xz_proj, vel_xz_proj)

            assert np.all(np.abs([az_ang, el_ang]) <= np.pi), "Angles should be in the range [-pi, pi]"

            new_state = dict()
            # Normalize obs to be in the range [-1, 1] (from [-pi, pi])
            new_state["azimuth_angle"] = np.array([az_ang/np.pi])
            new_state["elevation_angle"] = np.array([el_ang/np.pi])
            new_state["ang_pos"] = np.array([ang_pos/np.pi])
            new_state["ang_vel"] = np.array([ang_vel/np.pi])
            new_state["altitude"] = np.array([lin_pos[2] / self.flight_dome_size])  # Normalize altitude to be in the range [0, 1]

            # Store non-observable states (for debugging/evaluation purposes)
            self.info["aux_state"] = super().compute_auxiliary()
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
        self.reward, components = self.reward_func.yield_reward(self.state, self.action)

        self.info["reward_components"] = {
            "w_los_reward": components["los_reward"]["unweighted"],  # weighted LOS reward
            "w_los_smooth_reward": components["smooth_reward"]["unweighted"],  # weighted smooth reward
        }

        """Handle termination, truncation, and reward specifically for single waypoint."""
        # collision
        if np.any(self.env.contact_array[self.env.planeId]):
            self.reward = self.crash_reward
            self.info["collision"] = True
            self.termination |= True

        # exceed flight dome
        if np.linalg.norm(self.env.state(0)[-1]) > self.flight_dome_size:
            self.info["out_of_bounds"] = True
            self.termination |= True

        # unstable flight
        if np.any(np.abs(self.state["ang_pos"]) > 0.6*np.pi):
            self.reward = self.unstable_reward
            self.info["unstable"] = True
            self.termination |= True

        # target reached
        # if self.waypoints.target_reached:
        if self.info["distance_to_target"] < self.goal_reach_distance:
            self.reward = self.reached_reward
            self.waypoints.advance_targets()
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] = self.waypoints.num_targets_reached

    def get_info(self):
        return self.info
