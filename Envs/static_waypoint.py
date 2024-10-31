from PyFlyt.gym_envs.quadx_envs.quadx_waypoints_env import QuadXWaypointsEnv
import gymnasium
import numpy as np
from PyFlyt.gym_envs.utils.waypoint_handler import WaypointHandler
from gymnasium import spaces

class SingleWaypointQuadXEnv(QuadXWaypointsEnv):
    def __init__(self, **kwargs):
        # Force only one waypoint in the environment
        super().__init__(num_targets=1, sparse_reward=True, **kwargs)

        # the whole implicit state space = attitude + previous action + auxiliary information
        self.combined_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                10  # not 12 because we do not consider linear position
                + self.action_space.shape[0]
                + self.auxiliary_space.shape[0],
            ),
            dtype=np.float64,
        )

        # Redefine the observation space to remove the sequence observation
        self.observation_space = gymnasium.spaces.Dict({
            "attitude": self.combined_space,
            "target_deltas": gymnasium.spaces.Box(
                low=-2 * self.flight_dome_size,
                high=2 * self.flight_dome_size,
                shape=(3,),
                dtype=np.float64,
            ),
            "altitude": gymnasium.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float64,
            )
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
            altitude = lin_pos[2] if lin_pos[2] < 1.0 else 1.0
            self.state["attitude"] = np.concatenate([
                ang_vel, quaternion, lin_vel, self.action, super().compute_auxiliary()
            ], axis=-1)
            self.state["target_deltas"] = self.waypoints.distance_to_targets(ang_pos=ang_pos, lin_pos=lin_pos, quaternion=quaternion)[0]
            self.state["altitude"] = np.array([altitude])
        else:
            raise NotImplementedError("Only quaternion representation is supported for now.")

    def reset(self, *, seed=None, options=None):
        """Reset the environment with only one waypoint."""
        return super().reset(seed=seed, options=options)

    def compute_term_trunc_reward(self):
        """Handle termination, truncation, and reward specifically for single waypoint."""
        super().compute_term_trunc_reward()
        distance_reward = -np.linalg.norm(self.state["target_deltas"])  # Negative distance to target as reward
        altitude_reward = -1/self.state["altitude"]  # Negative altitude as reward
        self.reward = distance_reward + altitude_reward



