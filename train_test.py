import datetime
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import PyFlyt.gym_envs
import Envs.register

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from custom_callbacks import ObservationHistCallback

# Logdir
dir = "./tensorboard_log/StaticWaypointEnv/SimpleObs"
# hist_callback = ObservationHistCallback()
# Get current date and time
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
video_folder_path = f"./videos/run_{current_time}"

# Parallel environments
# vec_env = gym.make("Quadx-Waypoint-v0")
vec_env = make_vec_env("Quadx-Waypoint-v0", n_envs=1)
# vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

# model = PPO("MlpPolicy", vec_env, verbose=1, tensorboard_log=dir)  # For non-dict observation space
model = PPO("MultiInputPolicy", vec_env, verbose=1, tensorboard_log=dir)  # For non-dict observation space
model.learn(total_timesteps=500_000)
model.save("ppo_waypoint")
