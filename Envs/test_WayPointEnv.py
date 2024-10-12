import random

import gymnasium
import PyFlyt.gym_envs
import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

# env = gymnasium.make("Quadx-Waypoint-v0", render_mode="human")
env = gymnasium.make("Quadx-Waypoint-v0", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()

while not (term or trunc):
    new_action = env.action_space.sample()
    obs, rew, term, trunc, _ = env.step(new_action)
    print(f'Current reward: {rew}')