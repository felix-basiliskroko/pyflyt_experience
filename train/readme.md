# RL Training for UAV Navigation

## Overview

This repository provides scripts to train reinforcement learning (RL) models for UAV navigation using Proximal Policy Optimization (PPO), Soft Actor-Critic (SAC), and Deep Deterministic Policy Gradient (DDPG). The training environment is designed for UAV flight control, supporting different flight modes and hyperparameter configurations.

## Prerequisites

Ensure you have Python 3 installed along with the required dependencies. You can install the dependencies using:

```sh
pip install torch argparse
```

If your RL algorithms rely on additional packages like Stable Baselines3 or Gym, install them as well:

```sh
pip install stable-baselines3 gym
```

## Running Training

To train a model, use the `main.py` script with appropriate command-line arguments.

### Example Usage

```sh
python3 main.py --algorithm="ppo" --total_steps=500000 --eval_freq=20000 \
                --hyperparam_mode="tuned" --flight_mode=1 --run_name="TestRun"
```

### All arguments

```plaintext
--algorithm       (str)   RL algorithm (ppo, sac, ddpg). (REQUIRED)
--total_steps     (int)   Total number of training steps. (REQUIRED)
--hyperparam_mode (str)   Hyperparameter configuration (default, tuned). (REQUIRED)
--flight_mode     (int)   Flight mode (1 for angular control, -1 for thrust control). (REQUIRED)
--run_name        (str)   Name of the run. (REQUIRED)
--num_runs        (int)   Number of training runs (default: 1).
--eval_freq       (int)   Frequency of evaluations (default: 20000).
--shift           (float) Shift applied to the reward function (default: 0.75).
--lr              (float) Learning rate (default varies per algorithm).
--lr_mode         (str)   Learning rate schedule (constant, linear, exponential, cosine).
--log_root_dir    (str)   Directory for logging (default: ../logs/new_runs).
--check_root_dir  (str)   Directory for saving checkpoints (default: ../checkpoints/new_runs).
```


## Customizing Training

### Adjusting Policy Architecture

The policy network architecture is defined in `main.py`:

```python
if args.algorithm == "ppo":
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], vf=[64, 64]))
else:  # SAC/DDPG
    policy_kwargs = dict(activation_fn=t.nn.Tanh, net_arch=dict(pi=[64, 64], qf=[64, 64]))
```

Modify `net_arch` to change the layer sizes of the policy and value networks.

### Changing the Environment

By default, training occurs in `SingleWaypointQuadXEnv-v0`. Modify this value in `main.py` if using a different environment:

```python
env_id="SingleWaypointQuadXEnv-v0"
```

## Saving and Logging

- **TensorBoard Logs**: Training logs are saved in `../logs/new_runs` by default. You can visualize them using:

  ```sh
  tensorboard --logdir=../logs/new_runs
  ```

- **Model Checkpoints**: Saved in `../checkpoints/new_runs`. You can load a trained model for evaluation.

## Notes

- Ensure the appropriate environment is registered and compatible with the chosen RL algorithm.
- Modify `train_ppo.py`, `train_sac.py`, and `train_ddpg.py` for fine-grained control over training hyperparameters.

---

Happy training!
