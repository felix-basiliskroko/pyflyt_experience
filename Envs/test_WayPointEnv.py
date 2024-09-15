import random

import gymnasium
import PyFlyt.gym_envs
import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

env = gymnasium.make("Quadx-Waypoint-v0", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
print(f'First observation: {obs}')
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())