from ppo_tuning import entr_tune, gamma_tune, gae_tune, lr_tune


env_id = "SingleWaypointQuadXEnv-v0"
log_dir = "../../logs/hyperparam_log"

# entr_tune(env_id, log_dir, value_range=[0.0, 0.1], buckets=10, num_steps=75_000)
# gamma_tune(env_id, log_dir, value_range=[0.8, 0.99], buckets=10, num_steps=75_000)
# gae_tune(env_id, log_dir, value_range=[0.9, 1.0], buckets=5, num_steps=75_000)
lr_tune(env_id, log_dir, value_range=[1e-5, 5e-3], buckets=7, num_steps=75_000)
