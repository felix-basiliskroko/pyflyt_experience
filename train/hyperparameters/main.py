from ppo_tuning import entr_tune, gamma_tune, gae_tune, lr_tune, ppo_tune
from ppo_tuning import batch_tune, entr_tune, gamma_tune, gae_tune, lr_tune, ppo_tune
import os


env_id = "SingleWaypointQuadXEnv-v0"
log_dir = "./logs/hyper_ppo"

# hyperparameters_dir = "hyperparameters_ppo.json"

# Appropriate Ranges assumed from https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbe
# batch_tune(env_id, log_dir, values=[64, 128, 256, 512, 1024, 2048], num_steps=300_000)
# gamma_tune(env_id, log_dir, value_range=[0.93, 0.99], buckets=5, num_steps=300_000)
# entr_tune(env_id, log_dir, value_range=[0.0, 0.01], buckets=5, num_steps=300_000)
# gae_tune(env_id, log_dir, value_range=[0.92, 1.0], buckets=5, num_steps=300_000)
lr_tune(env_id, log_dir, value_range=[1e-5, 5e-3], buckets=5, num_steps=300_000)

'''
if not os.path.exists(hyperparameters_dir):
    print(f"File does not exist: {hyperparameters_dir}")

ppo_tune(env_id=env_id, log_dir=log_dir, buckets=5, num_buffer_steps=2048, num_total_steps=60_000, n_runs=5,
         n_envs=1, n_eval_eps=250, hyperparameter_save_path="hyperparameters_ppo.json",
         ent_coeffs=[0.0, 0.1], gammas=[0.85, 0.99], gae_lambdas=[0.9, 1.0], learning_rates=[1e-5, 5e-3])
         '''
