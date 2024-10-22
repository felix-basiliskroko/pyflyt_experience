from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register


env = gym.make("Quadx-Waypoint-v0", render_mode="human")
model = PPO("MultiInputPolicy", env=env)
model.load("ppo_waypoint")

term, trunc = False, False
obs, info = env.reset()
ep_reward = 0.0

for _ in range(1):
    # Evaluate the agent
    while not (term or trunc):
        action, _ = model.predict(obs, deterministic=False)
        action = action.squeeze(0)
        print(f"Action: {action.shape}")
        obs, rew, term, trunc, _ = env.step(action)
        ep_reward += rew
    print(f'Episode reward: {ep_reward}')
    env.reset()