# from ppo_tuning import batch_tune, entr_tune, gamma_tune, gae_tune, lr_tune
# from sac_tuning import batch_tune, buffersize_tune, action_noise_tune, learning_starts_tune, entr_tune, lr_tune
from ddpg_tuning import batch_tune, action_noise_tune, lr_tune
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import os

env_id = "SingleWaypointQuadXEnv-v0"
log_dir = "./logs/hyper_ddpg"

# hyperparameters_dir = "hyperparameters_ppo.json"

# Appropriate Ranges assumed from https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
# batch_tune(env_id, log_dir, values=[64, 128, 256, 512, 1024, 2048], num_steps=300_000)
# gamma_tune(env_id, log_dir, value_range=[0.93, 0.99], buckets=5, num_steps=300_000)
# entr_tune(env_id, log_dir, value_range=[0.0, 0.01], buckets=5, num_steps=300_000)
# gae_tune(env_id, log_dir, value_range=[0.92, 1.0], buckets=5, num_steps=300_000)
# lr_tune(env_id, log_dir, value_range=[1e-5, 5e-3], buckets=5, num_steps=300_000)

# batch_tune(env_id, log_dir, values=[128, 256, 512, 1024], num_steps=600_000)
# buffersize_tune(env_id, log_dir, values=[1_000_000, 1_500_000, 2_000_000], num_steps=400_000)
'''
action_noise_tune(env_id, log_dir,
                  values=[NormalActionNoise(mean=np.array([0.0, 0.0, 0.0, 0.0]),
                                                  sigma=np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5])),
                          OrnsteinUhlenbeckActionNoise(mean=np.array([0.0, 0.0, 0.0, 0.0]),
                                                       sigma=np.array([0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5])),
                          None],
                  num_steps=600_000)
'''
# learning_starts_tune(env_id, log_dir, values=[1_000, 5_000, 10_000], num_steps=400_000)
# entr_tune(env_id, log_dir, value_range=[0.005, 0.01], buckets=4, num_steps=400_000)
lr_tune(env_id, log_dir, value_range=[3e-5, 1e-4], buckets=5, num_steps=400_000)