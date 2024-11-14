import gymnasium
import PyFlyt.gym_envs

env_id = "SingleWaypointQuadXEnv-v0"
env = gymnasium.make(env_id, render_mode=None)

term, trunc = False, False
obs, _ = env.reset()
print(f'First observation: {obs}')
while not (term or trunc):
    obs, rew, term, trunc, _ = env.step(env.action_space.sample())