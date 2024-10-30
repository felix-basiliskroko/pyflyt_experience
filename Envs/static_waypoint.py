from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
import gymnasium
import numpy as np
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler


class SingleWaypointQuadXEnv(QuadXWaypointsEnv):
    def __init__(self, **kwargs):
        # Force only one waypoint in the environment
        super().__init__(num_targets=1, **kwargs)

        # Redefine the observation space to remove the sequence observation
        self.observation_space = gymnasium.spaces.Dict({
            "attitude": self.combined_space,
            "target_deltas": gymnasium.spaces.Box(
                low=-2 * self.flight_dome_size,
                high=2 * self.flight_dome_size,
                shape=(3,),
                dtype=np.float64,
            ),
        })

        # define waypoints
        self.waypoints.flight_dome_size = 10.0
        self.waypoints.goal_reach_distance = 0.5
        self.waypoints.enable_render = True
        self.waypoints.min_height = 5

    def compute_state(self):
        """Computes the state for a single waypoint environment."""
        super().compute_state()
        # Since there's only one waypoint, we take the first and only target delta
        if self.angle_representation == 1:
            ang_vel, ang_pos, lin_vel, lin_pos, quaternion = super().compute_attitude()
            self.state["attitude"] = np.concatenate([
                ang_vel, quaternion, lin_vel, lin_pos, self.action, super().compute_auxiliary()
            ], axis=-1)
            self.state["target_deltas"] = self.waypoints.distance_to_targets(ang_pos=ang_pos, lin_pos=lin_pos, quaternion=quaternion)[0]
        else:
            raise NotImplementedError("Only quaternion representation is supported for now.")

    def reset(self, *, seed=None, options=None):
        """Reset the environment with only one waypoint."""
        return super().reset(seed=seed, options=options)

    def compute_term_trunc_reward(self):
        """Handle termination, truncation, and reward specifically for single waypoint."""
        super().compute_term_trunc_reward()

        if self.waypoints.target_reached:
            # Reward adjustments and handling when a target is reached.
            self.reward += 100.0  # Arbitrary reward for reaching the waypoint.
            self.waypoints.advance_targets()  # Advances to the next target, if any.

            # Update info and done status
            self.truncation |= self.waypoints.all_targets_reached
            self.info["env_complete"] = self.waypoints.all_targets_reached
            self.info["num_targets_reached"] += 1
