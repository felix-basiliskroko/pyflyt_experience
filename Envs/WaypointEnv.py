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

        self.sparse_reward = sparse_reward
        self.use_yaw_target = use_yaw_target
        self.goal_reach_distance = goal_reach_distance
        self.goal_reach_angle = goal_reach_angle

        # This defines the adapted observation space for the Waypoint environment
        self.observation_space = spaces.Dict({
            "attitude": spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64),
            "prev_action": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "auxiliary": spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64),
            "target_delta": spaces.Box(low=-4 * flight_dome_size, high=4 * flight_dome_size, shape=(3,),
                                       dtype=np.float64),  # Shape: 3,
            "previous_dist": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float64)
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
            "attitude": np.zeros(16),
            "prev_action": np.zeros(4),
            "auxiliary": np.zeros(4),
            "target_delta": np.zeros(3),
            "previous_dist": np.zeros(1)
        }

        super().end_reset()
        self.compute_state()

        return self.state, self.info

    def set_new_waypoint(self):
        r = np.random.uniform(0, self.adj_dome_size)
        theta = np.random.uniform(0, np.pi)
        phi = np.random.uniform(0, 2 * np.pi)

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        self.waypoint = np.array([x, y, z])

    def compute_state(self):
        """Compute the state of the QuadX."""
        ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()  # TODO: Insert distance to pursuer here as well."
        aux_state = super().compute_auxiliary()

        attitidue = np.concatenate([ang_vel, ang_pos, lin_vel, lin_pos, quaternion])
        target_delta = self.compute_target_delta(ang_pos, lin_pos, quaternion)

        '''
        print(f'Shape of ang_vel: {ang_vel.shape}')
        print(f'Shape of ang_pos: {ang_pos.shape}')
        print(f'Shape of lin_vel: {lin_vel.shape}')
        print(f'Shape of lin_pos: {lin_pos.shape}')
        print(f'Shape of quaternion: {len(quaternion)}')
        print(f'Shape of attitude: {attitidue.shape}')
        print("------------------------------------")
        print(f'Shape of auxiliary: {aux_state.shape}')
        print(f'Shape of target_delta: {target_delta.shape}')
        print(f'Shape of action: {self.action.shape}')'''

        # This is done, because self.state is 'None' in the beginning, accessing "target_delta" would throw an error.
        try:
            self.previous_distance = np.linalg.norm(self.state["target_delta"])
        except TypeError:
            self.previous_distance = self.initial_distance  # This yields a reward (distance_change_norm) of 0.0 in the first step.

        self.distance_change_norm = (self.previous_distance - np.linalg.norm(target_delta)) / self.initial_distance

        # Combine attitude, auxiliary information, target delta into the state dictionary
        self.state["attitude"] = np.array([attitidue], dtype=np.float64)
        self.state["prev_action"] = np.array([self.action], dtype=np.float64)
        self.state["auxiliary"] = np.array([aux_state], dtype=np.float64)
        self.state["target_delta"] = np.array([target_delta], dtype=np.float64)
        self.state["previous_dist"] = np.array([self.previous_distance], dtype=np.float64)

    def compute_base_term_trunc_reward(self) -> None:
        self.reward = self.distance_change_norm
        super().compute_base_term_trunc_reward()  # Overrides self.reward/self.termination if out_of_bounds, max_timesteps or collision

    def compute_target_delta(self, ang_pos, lin_pos,
                             quaternion):  # TODO: Consider adding ang_pos, quaternion to the as different options for the delta calculation.
        """Compute the delta to the waypoint."""
        if self.use_yaw_target:  # TODO: Look into what this specifically does
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
