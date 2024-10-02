"""QuadX Waypoints Environment."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium import spaces

from PyFlyt.gym_envs.quadx_envs.quadx_base_env import QuadXBaseEnv
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
            flight_mode: int = 0,
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

        self.waypoint = np.zeros(3)  # gets set in reset
        self.adj_dome_size = self.flight_dome_size if self.flight_dome_size < np.inf else 20.0  # TODO: Remove Hard Code: Add parameter in WayPointEnv to set the flight_dome_size.

        # Override the Observation space as defined in the superclass:
        self.attitude_space = spaces.Box(
            low=-self.adj_dome_size, high=self.adj_dome_size, shape=(3,), dtype=np.float64
        )
        self.auxiliary_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
        )

        self.sparse_reward = sparse_reward
        self.use_yaw_target = use_yaw_target
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle

        # Define observation space
        waypoint_space_shape = (4,) if use_yaw_target else (3,)
        self.observation_space = spaces.Dict({
            "attitude": self.combined_space,
            "target_delta": spaces.Box(low=-4 * flight_dome_size, high=4 * flight_dome_size, shape=waypoint_space_shape,
                                       dtype=np.float64),
        })

    def reset(self, *, seed: None | int = None, options: dict[str, Any] | None = dict()) -> tuple[
        dict[str, np.ndarray], dict[str, Any]]:
        """Reset the environment for a new episode."""

        self.set_new_waypoint()

        super().begin_reset(seed, options)
        super().end_reset()
        self.compute_state()

        # Evaluate what part of the state is responsible for the "not within observation space" error.
        attitude = self.state["attitude"]
        print(f'Attitude: {attitude}')

        return self.state, self.info

    def set_new_waypoint(self):
        r = np.random.uniform(0, self.adj_dome_size)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        way_point = np.array([x, y, z])

        return way_point

    def compute_state(self):
        """Compute the state of the QuadX."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()  # TODO: Insert distance to pursuer here as well."
        aux_state = super().compute_auxiliary()

        attitidue = np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos, self.action, aux_state])
        target_delta = self.compute_target_delta(ang_pos, lin_pos, quaternion)

        # Combine attitude and target delta into the state dictionary
        new_state = {
            "attitude": attitidue,
            "target_delta": target_delta,
        }
        self.state = new_state

    def compute_target_delta(self, ang_pos, lin_pos, quaternion):  #TODO: Consider adding ang_pos, quaternion to the as different options for the delta calculation.
        """Compute the delta to the waypoint."""
        if self.use_yaw_target: #TODO: Look into what this specifically does
            pass
        target_delta = self.waypoint[:3] - lin_pos
        return target_delta

    def compute_term_trunc_reward(self):
        """Computes the termination, truncation, and reward based on the current state."""
        super().compute_base_term_trunc_reward()
        distance_to_waypoint = np.linalg.norm(self.state["target_delta"])

        # Check if the waypoint has been reached within the specified tolerance
        if distance_to_waypoint <= self.goal_reach_distance:
            self.reward = 100.0 if self.sparse_reward else self.reward + 1000.0
            self.termination = True
        elif not self.sparse_reward:
            # Continuous reward for making progress towards the waypoint
            self.reward += 1.0 / distance_to_waypoint
