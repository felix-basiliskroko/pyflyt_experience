import gymnasium
import PyFlyt.gym_envs

env = gymnasium.make("PyFlyt/QuadX-Waypoints-v2", render_mode="human")

term, trunc = False, False
obs, _ = env.reset()
print(f'First observation: {obs}')
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())