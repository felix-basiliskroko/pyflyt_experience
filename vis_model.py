from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
import gymnasium as gym
from Envs.WaypointEnv import QuadXWaypoint
from Envs import register
import matplotlib.pyplot as plt
import numpy as np
# from stable_baselines3.common.evaluation import evaluate_policy
from Evaluation.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecMonitor, is_vecenv_wrapped
from stable_baselines3.common.monitor import Monitor


def plot_thrust_curve(model, vec_env):
    episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, vec_env, n_eval_episodes=1, render=render_m, deterministic=True, return_episode_rewards=True)
    thrusts = []
    ep_thrust = []

    trajectories = []

    # Iterate through each sublist (each trajectory)
    for sublist in all_obs:
        trajectory = []
        for record in sublist:
            # Extract the fourth element of the aux_state array
            print(record)
            aux_state_fourth_element = record['aux_state'][0][3]  # Assumes aux_state is a 2D array and we need the 4th element
            trajectory.append(aux_state_fourth_element)
        trajectories.append(trajectory)
        trajectory = []

    # thrusts.append([obs["aux_state"][3] for obs in ep])
    # Create a figure and an axes.
    plt.figure(figsize=(8, 6))

    # Plot each sublist
    for index, sublist in enumerate(trajectories):
        plt.plot(sublist, label=f'Series {index + 1}')

    # Add a legend
    plt.legend()

    # Add title and labels
    plt.title('Thrusts over time')
    plt.xlabel('timestep')
    plt.ylabel('thrust-value')

    # Show the plot
    plt.show()




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
        ax.plot(x[:i + 1], y[:i + 1], z[:i + 1], color='blue', alpha=alpha)

    # Plot the target point
    ax.scatter(target[0], target[1], target[2], color='red', s=100, label='Target')

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.legend()

    plt.show()


def vis_model(env_id="SingleWaypointQuadXEnv-v0",
              model_path="./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted-NoOutOfBoundsError/best_model"):
    env = gym.make(env_id, render_mode=None)
    model = PPO("MultiInputPolicy", env=env)
    model.load(model_path, deterministic=True)
    agent_pos = []

    term, trunc = False, False
    obs, info = env.reset()
    ep_reward = 0

    for _ in range(1):
        # Evaluate the agent
        while not (term or trunc):
            action, _ = model.predict(obs, deterministic=True)
            # action = action.squeeze(0)
            obs, rew, term, trunc, _ = env.step(action)
            print(f'Observation: {obs}')
            # info_state = env.get_info_state()
            # print(f'Current position: {info_state["lin_pos"]}; distance to target: {np.linalg.norm(info_state["lin_pos"] - env.waypoint)}')
            # print(f'Action taken: {action}, with reward: {rew}')
            # print(f'Current waypoint: {info_state["lin_pos"]}')
            # print(f'Current velocity: {info_state["lin_vel"]/np.linalg.norm(info_state["lin_vel"])}')
            # agent_pos.append(info_state['lin_pos'])

            ep_reward += rew

        print(f'Episode reward: {ep_reward}')
        env.reset()


#  ---------------------------------------------------------------------------------------------------------------------

run_path = "./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/tmp"
model_path = run_path + "/best_model"
eval_file_path = run_path + "/evaluations.npz"
env_id = "SingleWaypointQuadXEnv-v0"

deterministic = True
render = None  # #None
render_m = False
num_eval_eps = 1

print(f'Over {num_eval_eps} episodes, with deterministic set to {deterministic}:')

print("--------------------------- evaluation.npz ----------------------------------")

evaluations = np.load(eval_file_path)
print(f'Mean reward: {np.mean(evaluations["results"][1], axis=0)}, with standard deviation: {np.std(evaluations["results"][1], axis=0)}')
print(f'Mean episode lengths: {np.mean(evaluations["ep_lengths"][1], axis=0)}, with standard deviation: {np.std(evaluations["ep_lengths"][1], axis=0)}')

print("-----------------------------------------------------------------------------")
print("---------------------------- evaluate_policy --------------------------------")

# vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=42)
vec_env = gym.make(env_id, render_mode=render)
# print(f'Wrapped in Monitor: {is_vecenv_wrapped(vec_env, VecMonitor) or vec_env.env_is_wrapped(Monitor)[0]}')
observations = vec_env.reset()

model = PPO.load(model_path, deterministic=deterministic)
plot_thrust_curve(model, vec_env)

# episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, vec_env, n_eval_episodes=num_eval_eps, render=render_m, deterministic=deterministic, return_episode_rewards=True)
print(f'Amount of information: {len(all_infos)}')
print(f'Information: {all_infos}')
mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

print(f'Rewards: {episode_rewards}')
print(f'Episode lengths: {episode_lengths}')
print(f'Mean reward: {mean_reward}, Standard deviation of reward: {std_reward}')
print(f'Mean episode length: {mean_ep_length}, Standard deviation of episode length: {std_ep_length}')

print("-----------------------------------------------------------------------------")

#  ---------------------------------------------------------------------------------------------------------------------
