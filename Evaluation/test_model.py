from stable_baselines3 import PPO
import PyFlyt.gym_envs
from stable_baselines3.common.env_util import make_vec_env

# Recreate the environment
vec_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v2", n_envs=4)
model = PPO.load("ppo_cartpole")
obs = vec_env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = vec_env.step(action)
    vec_env.render("human")
