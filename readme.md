# PyFlyt Experience - Debug Convergence Branch

*A Reinforcement Learning Approach for UAV Waypoint Navigation*

## Overview

This repository contains the **debug\_convergence** branch of my research on UAV waypoint navigation using reinforcement learning (RL). The project explores the convergence behavior of various RL algorithms within a tailored PyFlyt simulation environment. The primary focus is on stabilizing training dynamics, optimizing hyperparameters, and refining the reward function for improved UAV control.

## Project Scope

The objective of this project is to apply **Proximal Policy Optimization (PPO)**, **Soft Actor-Critic (SAC)**, and **Deep Deterministic Policy Gradient (DDPG)** for UAV waypoint navigation in a simulated environment. The work builds on insights from my bachelor thesis:

> *Analysis of Various Reinforcement Learning Algorithms for Unmanned Aerial Vehicle Waypoint Navigation Approaches*\
> [Technische Hochschule Ingolstadt, 2025]

## Features

- **Custom PyFlyt Gymnasium Environment**: Extended state and action spaces, including angular positional data and reward function refinements.
- **Reinforcement Learning Algorithms**: Implementation of PPO, SAC, and DDPG for UAV control.
- **Hyperparameter Optimization**: Fine-tuned learning rates, entropy coefficients, and discount factors.
- **Debugging Convergence**: Analysis of training stability, sample efficiency, and model robustness.

## Repository Structure

```
pyflyt_experience/
 ├── Envs/     # Contains the custom PyFlyt Gymnasium environment and Reward function
 ├── Evaluation/     # Contains scripts and notebooks for evaluation of the trained models
 ├── VisualsBA/           # Contains the scripts and plots for the visualizations of the results (used in my thesis)
 ├── checkpoints/          # Contains the model checkpoints saved during training
 ├── logs/          # Contains the tensorboard logs for the training of the models
 ├── models/           # Pre-trained RL models, for angular and thrust control
 ├── requirements/          # Contains two requirements files, one for devices with CUDA and one for devices without CUDA support
 ├── train/          # Contains the training scripts for the RL models as well as the tensorboard logs for hyperparameter optimization
 ├── README.md         # This document
```

## Results

Experiments indicate that **PPO outperforms SAC and DDPG** in terms of:

- **Training stability**
- **Sample efficiency**
- **Generalization across waypoint scenarios**

See the full analysis in the *results* directory or refer to my thesis for in-depth evaluations.

## Installation

```bash
# Clone the repository
git clone --branch debug_convergence https://github.com/felix-basiliskroko/pyflyt_experience.git
cd pyflyt_experience

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

\#\# Usage

### Train a model

```bash
python scripts/train.py --algorithm PPO --env QuadXWaypointsEnv
```

### Evaluate a trained model

```bash
python scripts/evaluate.py --model models/ppo_model.pth --env QuadXWaypointsEnv
```

## Key Insights

- **Observation Space Design**: The addition of angular velocity data enhances UAV stability.
- **Reward Function Optimization**: Combining Line-of-Sight (LOS) penalties with smooth control incentives leads to more efficient navigation.
- **Training Dynamics**: Learning rate annealing significantly improves convergence.

## License

This project is licensed under the MIT License. See `LICENSE` for details.

## Acknowledgments

- **Technische Hochschule Ingolstadt** for academic support.
- **Airbus Defence and Space** for collaboration in UAV simulation frameworks.
- **PyFlyt Developers** for providing an extensible UAV simulation environment.

## Contact

Felix Unterleiter\
[GitHub Profile](https://github.com/felix-basiliskroko)\
[LinkedIn](https://your-linkedin.com)

---

**Note:** This README refers exclusively to the **debug\_convergence** branch, which is the main focus of ongoing development.

