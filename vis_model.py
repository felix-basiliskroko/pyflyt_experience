from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_with_target(trajectory_points, target):
    """
    Plot a trajectory in 3D space from a list of 3D points, with increasing visibility, and a target point.

    :param trajectory_points: List of numpy arrays, each array is of shape (3,) representing a point in 3D space.
    :param target: A numpy array of shape (3,) representing the target point in 3D space.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for plotting
    x, y, z = zip(*trajectory_points)  # Unpack points into separate coordinate lists

    # Plot trajectory with increasing visibility
    for i in range(len(trajectory_points)):
        alpha = i / len(trajectory_points) * 0.9 + 0.1  # Gradually increase visibility
        ax.plot(x[:i+1], y[:i+1], z[:i+1], color='blue', alpha=alpha)

    # Plot the target point
    ax.scatter(target[0], target[1], target[2], color='red', s=100, label='Target')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    plt.show()


env = gym.make("Quadx-Waypoint-v0", render_mode=None)
model = PPO("MultiInputPolicy", env=env)
model.load("./checkpoints/SimpleObs/Altitude-Reward-New-Action-Space/best_model", deterministic=True)
agent_pos = []

term, trunc = False, False
obs, info = env.reset()
ep_reward = 0.0
pitch_ctrl = 0.
yaw_ctrl = 0.
yaw_step = 0.1
ct = 0

for _ in range(1):
    # Evaluate the agent
    while not (term or trunc):
        # action, _ = model.predict(obs, deterministic=False)
        # action = action.squeeze(0)
        if abs(yaw_ctrl) >= 1*np.pi:
            yaw_step *= -1
        yaw_ctrl += yaw_step

        action = np.array([0.5, 0., 0., .45])

        # action = np.array([0.8, 0.8, 0.8, 0.8])
        obs, rew, term, trunc, _ = env.step(action)
        info_state = env.get_info_state()
        print(f'Current angular position: {info_state["ang_pos"]} at yaw_ctrl: {yaw_ctrl}; Amplitude of angular_velocities: {np.linalg.norm(info_state["ang_vel"])}')
        # print(f'Current waypoint: {info_state["lin_pos"]}')
        # print(f'Current velocity: {info_state["lin_vel"]/np.linalg.norm(info_state["lin_vel"])}')
        agent_pos.append(info_state['lin_pos'])

        ep_reward += rew

        ct += 1
        if ct == 1000:
            break

    print(f'Episode reward: {ep_reward}')
    env.reset()

plot_trajectory_with_target(agent_pos, env.waypoint)