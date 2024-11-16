from ppo_tuning import entr_tune


env_id = "SingleWaypointQuadXEnv-v0"
log_dir = "../../logs/hyperparam_log"

entr_tune(env_id, log_dir, value_range=[0.0, 0.1], buckets=10, num_steps=75_000)
