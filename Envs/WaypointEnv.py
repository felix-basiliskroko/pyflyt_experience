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
            flight_mode: int = -1,  # This needs to be set to -1 to use thrust control between (0, 0.8) -> quadx_base_env.py
            flight_dome_size: float = 5_000.0,
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
        self.max_speed = 50.0  # This is approximated leaving the quadx on full thrust (on all motors) for 200 steps and recording the max speed  # TODO: Somewhere inside the source code there should be some real value, not just some arbitrary number
        self.normaliser = Normaliser(alpha=0.7, max_speed=self.max_speed, border_radius=2*flight_dome_size)

        self.waypoint = np.zeros(3)  # gets set in reset

        self.sparse_reward = sparse_reward
        self.use_yaw_target = use_yaw_target
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle

        # This defines the adapted observation space for the Waypoint environment
        # ang_vel, ang_pos, lin_vel, lin_pos, quaternion
        self.observation_space = spaces.Dict({
            "ang_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "ang_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "lin_vel": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "lin_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
            "quaternion": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "prev_action": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "auxiliary": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "target_delta": spaces.Box(low=-4 * flight_dome_size, high=4 * flight_dome_size, shape=(3,),
                                       dtype=np.float64),  # Shape: 3,
        })

    def reset(self, *, seed: None | int = None, options: dict[str, Any] | None = dict()) -> tuple[
        dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode."""

        self.set_new_waypoint()
        # Save initial distance for scaled reward calculation
        self.initial_distance = np.linalg.norm(self.waypoint)

        super().begin_reset(seed, options)

        # Overwrite the state dictionary, instead of "None" as initialized in the super class
        self.state = {
            "ang_vel": np.zeros(3, dtype=np.float64),
            "ang_pos": np.zeros(3, dtype=np.float64),
            "lin_vel": np.zeros(3, dtype=np.float64),
            "lin_pos": np.zeros(3, dtype=np.float64),
            "quaternion": np.zeros(4, dtype=np.float64),
            "prev_action": np.zeros(4, dtype=np.float64),
            "auxiliary": np.zeros(4, dtype=np.float64),
            "target_delta": np.zeros(3, dtype=np.float64),
        }

        super().end_reset()
        self.compute_state()

        return self.state, self.info

    def set_new_waypoint(self):
        limit_perc = 0.1
        r = np.random.uniform(0, limit_perc * self.flight_dome_size)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.waypoint = np.array([x, y, z])

    def compute_state(self):
        """Compute the state of the QuadX."""
        # Compute observation
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
        aux_state = super().compute_auxiliary()
        target_delta = self.compute_target_delta(ang_pos, lin_pos, quaternion)

        # Normalise
        norm_state = self.normaliser.simple_normaliser(lin_pos=lin_pos,
                                                       lin_vel=lin_vel,
                                                       target_delta=target_delta,
                                                       prev_action=self.action,
                                                       aux_state=aux_state)

        # Adapt the state dictionary
        self.state["ang_vel"] = np.array([ang_vel], dtype=np.float64)
        self.state["ang_pos"] = np.array([ang_pos], dtype=np.float64)
        self.state["lin_vel"] = np.array([norm_state["lin_vel"]], dtype=np.float64)
        self.state["lin_pos"] = np.array([norm_state["lin_pos"]], dtype=np.float64)
        self.state["quaternion"] = np.array([quaternion], dtype=np.float64)
        self.state["prev_action"] = np.array([norm_state["prev_action"]], dtype=np.float64)
        self.state["auxiliary"] = np.array([norm_state["aux_state"]], dtype=np.float64)
        self.state["target_delta"] = np.array([norm_state["target_delta"]], dtype=np.float64)

    def compute_target_delta(self, ang_pos, lin_pos,
                             quaternion):  # TODO: Consider adding ang_pos, quaternion to the as different options for the delta calculation.
        """Compute the delta to the waypoint."""
        if self.use_yaw_target:  # TODO: Look into what this specifically does
            pass
        target_delta = self.waypoint[:3] - lin_pos
        return target_delta

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward based on the current state."""
        self.reward = -(np.linalg.norm(self.state["target_delta"]) * 20)**2

        super().compute_base_term_trunc_reward()  # This evaluates the termination and truncation criteria inside the super class
