"""QuadX Waypoints Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
from normaliser import Normaliser
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class QuadXWaypoint(QuadXBaseEnv):
    """QuadX Environment for navigating to a single waypoint.

    Args:
    ----
        waypoint: A single `[x, y, z, (optional) yaw]` waypoint.
        sparse_reward (bool): whether to use sparse rewards or not.
        use_yaw_target (bool): whether to match yaw target before the waypoint is considered reached.
        goal_reach_distance (float): distance to the waypoint for it to be considered reached.
        goal_reach_angle (float): angle in radians to the waypoint for it to be considered reached, only effective if `use_yaw_target` is used.
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
            sparse_reward: bool = False,
            use_yaw_target: bool = False,
            goal_reach_distance: float = 0.2,
            goal_reach_angle: float = 0.1,
            flight_mode: int = 0,  # This needs to be set to 0 to use pitch, yaw, roll, thrust_level -> quadx_base_env.py
            flight_dome_size: float = 5_000.0,
            spawn_point_scheduler: bool = False,
            max_duration_seconds: float = 10.0,
            angle_representation: Literal["euler", "quaternion"] = "quaternion",
            agent_hz: int = 30,
            render_mode: None | Literal["human", "rgb_array"] = None,
            render_resolution: tuple[int, int] = (480, 480),
    ):
        super().__init__(
            flight_mode=flight_mode,
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            angle_representation=angle_representation,
            agent_hz=agent_hz,
            render_mode=render_mode,
            render_resolution=render_resolution,
        )

        # Initialize the Normalizer
        self.distance_change_norm = None
        self.spawn_point_r = 0.2
        self.max_speed = 50.0  # This is approximated leaving the quadx on full thrust (on all motors) for 200 steps and recording the max speed  # TODO: Somewhere inside the source code there should be some real value, not just some arbitrary number
        # self.normaliser = Normaliser(alpha=0.7, max_speed=self.max_speed, border_radius=spawn_point_range * (2*flight_dome_size))

        self.waypoint = np.zeros(3)  # gets set in reset

        self.sparse_reward = sparse_reward
        self.use_yaw_target = use_yaw_target
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle
        self.info_state = {}
        self.prev_aux_state = np.zeros(4)

        # This defines the adapted observation space for the Waypoint environment
        # ang_vel, ang_pos, lin_vel, lin_pos, quaternion
        self.observation_space = spaces.Dict({
            "targ_delta": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "targ_distance": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "lin_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "altitude": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64),
            "auxiliary": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "angular_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "angular_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
        })

        '''self.observation_space = spaces.Dict({
            "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "ang_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "lin_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "lin_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "prev_action": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "auxiliary": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "target_delta": spaces.Box(low=-4 * flight_dome_size, high=4 * flight_dome_size, shape=(3,),
                                       dtype=np.float64),  # Shape: 3,
        })'''

        self.waypoint = np.array([20.0, 20.0, 20.0], dtype=np.float64)
        # Save initial distance for scaled reward calculation
        self.initial_distance = np.linalg.norm(self.waypoint)

    def reset(self, *, seed: None | int = None, options: dict[str, Any] | None = dict()) -> tuple[
        dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode."""

        super().begin_reset(seed, options)

        # Overwrite the state dictionary, instead of "None" as initialized in the super class
        '''self.state = {
            "ang_vel": np.zeros(3, dtype=np.float64),
            "ang_pos": np.zeros(3, dtype=np.float64),
            "lin_vel": np.zeros(3, dtype=np.float64),
            "lin_pos": np.zeros(3, dtype=np.float64),
            "quaternion": np.zeros(4, dtype=np.float64),
            "prev_action": np.zeros(4, dtype=np.float64),
            "auxiliary": np.zeros(4, dtype=np.float64),
            "target_delta": np.zeros(3, dtype=np.float64),
        }'''

        self.state = {
            "targ_delta": np.zeros(3, dtype=np.float64),
            "targ_distance": np.zeros(1, dtype=np.float64),
            "lin_vel": np.zeros(3, dtype=np.float64),
            "altitude": np.zeros(1, dtype=np.float64),
        }

        super().end_reset()
        self.compute_state()

        return self.state, self.info

    def set_new_waypoint(self, use_schedule=False):
        r = np.random.uniform(0, self.spawn_point_r * self.flight_dome_size)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.waypoint = np.array([x, y, z])

    def get_info_state(self):
        """
        Provides additional information, that are not part of the state-space.
        :return: dict
        """
        return self.info_state

    def step(self, action: np.ndarray) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Returns:
        -------
            state, reward, termination, truncation, info

        """
        # unsqueeze the action to be usable in aviary
        self.action = action.copy()

        # reset the reward and set the action
        self.reward = -0.1
        self.env.set_setpoint(0, action)

        # step through env, the internal env updates a few steps before the outer env
        for _ in range(self.env_step_ratio):
            # if we've already ended, don't continue
            if self.termination or self.truncation:
                break

            self.env.step()

            # compute state and done
            self.compute_state()
            self.prev_aux_state = self.state["auxiliary"].squeeze()
            self.compute_term_trunc_reward()

        # increment step count
        self.step_count += 1

        return self.state, self.reward, self.termination, self.truncation, self.info

    def compute_state(self):
        """Compute the state of the QuadX."""
        # Compute observation
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        self.aux_state = super().compute_auxiliary()
        target_delta = self.compute_target_delta(ang_pos=None, lin_pos=lin_pos, quaternion=None)
        norm_target_delta = target_delta / np.linalg.norm(target_delta)
        norm_targ_distance = np.linalg.norm(target_delta) / (2*self.initial_distance)
        # norm_altitude = lin_pos[2] / (self.spawn_point_r * self.flight_dome_size)
        norm_altitude = lin_pos[2] / 25  # this is because the static target is at 20m (+5 for leniency)
        #TODO: Update the normalisation of the altitude based on the height of the WAYPOINT (+ some leniency-factor L)

        # Provide addition information (for evaluation/plotting etc.)
        self.info_state = {
            "ang_vel": ang_vel,
            "ang_pos": ang_pos,
            "lin_vel": lin_vel,
            "lin_pos": lin_pos,
            "quaternion": quaternion
        }

        self.state["targ_delta"] = np.array([norm_target_delta], dtype=np.float64)
        self.state["targ_distance"] = np.array([norm_targ_distance], dtype=np.float64)
        self.state["lin_vel"] = np.array([lin_vel/np.linalg.norm(lin_vel)], dtype=np.float64)
        self.state["altitude"] = np.array([norm_altitude], dtype=np.float64)
        self.state["auxiliary"] = np.array([self.prev_aux_state], dtype=np.float64)
        self.state["angular_vel"] = np.array([ang_vel], dtype=np.float64)
        self.state["angular_pos"] = np.array([ang_pos], dtype=np.float64)


    def compute_target_delta(self, ang_pos, lin_pos,
                             quaternion):  # TODO: Consider adding ang_pos, quaternion to the as different options for the delta calculation.
        """Compute the delta to the waypoint."""
        if self.use_yaw_target:  # TODO: Look into what this specifically does
            pass
        target_delta = self.waypoint[:3] - lin_pos
        return target_delta

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward based on the current state."""
        distance_target_reward = -(self.state["targ_distance"])
        distance_ground_reward = -5 * (0.2 - self.state["altitude"]) if self.state["altitude"] < 0.2 else 0  # 5m above ground
        smooth_control_reward = -np.linalg.norm(self.prev_aux_state - self.action.copy())/np.sqrt(12*np.pi**2 + (16/25))
        smooth_ang_vel_reward = -np.linalg.norm(self.state["angular_vel"])/np.sqrt(3)/np.sqrt(3*np.pi**2)

        self.reward = 1/4 * distance_target_reward + 1/4 * distance_ground_reward + 1/4 * smooth_control_reward + 1/4 * smooth_ang_vel_reward
        self.info_state["reward"] = {
            "distance_target_reward": distance_target_reward,
            "distance_ground_reward": distance_ground_reward,
            "smooth_control_reward": smooth_control_reward,
            "smooth_ang_vel_reward": smooth_ang_vel_reward,
        }

        agent_lin_pos = self.info_state["lin_pos"]

        if self.step_count > self.max_steps:
            self.truncation |= True

        # if anything hits the floor, basically game over
        if np.any(self.env.contact_array[self.env.planeId]):
            self.reward = -500.0
            self.info["collision"] = True
            self.termination |= True

        # reached waypoint
        if np.linalg.norm(self.waypoint - agent_lin_pos) <= 5.0:
            self.reward = 5.0
            self.info["collision"] = True
            self.termination |= True
