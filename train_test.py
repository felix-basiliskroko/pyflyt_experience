import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
import Envs.register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Logdir
dir = "./tensorboard_log/StaticWaypintEnv-RescaleNorm"
# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_folder_path = f"./videos/run_{current_time}"

# Parallel environments
vec_env = gym.make("Quadx-Waypoint-v0", render_mode=None)
# vec_env = make_vec_env("PyFlyt/QuadX-Pole-Balance-v2", n_envs=16)
# vec_env = RecordVideo(vec_env, video_folder=video_folder_path, episode_trigger=lambda episode_id: episode_id % 20 == 0)  # Adjust trigger as needed

# model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=dir)  # For non-dict observation space
model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=dir)  # For non-dict observation space
model.learn(total_timesteps=5_000_000)
model.save("ppo_waypoint")
