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


def plot_eval(results, var_name):
    """
    Plot the evaluation results.

    :param results: (dict) The evaluation results
    """

    assert len(results.keys()) == 1, 'Only one variable can be plotted at a time.'

    plt.figure(figsize=(8, 6))

    for index, sublist in enumerate(results[var_name]):
        plt.plot(sublist, color="blue")

    plt.legend()

    # Add title and labels
    plt.title(f'{var_name} over time')
    plt.xlabel('timestep')
    plt.ylabel(f'{var_name}-value')

    # Show the plot
    plt.show()


def aggregate_eval(model, env, n_eval_episodes, var_name):
    '''
    Evaluate the model on the environment for a given number of episodes and aggregate the values of a given variable.
    :param model: PPO Model
    :param env: Vectorized environment
    :param n_eval_episodes: Number of episodes to evaluate the model on
    :param var_name: Name(s) of the variable(s) to aggregate
    :return: Dictionary containing the aggregated values of the variable(s)
    '''
    assert type(var_name) == str or type(var_name) == list, 'Variable name must be a string or a list of strings.'

    episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, vec_env, n_eval_episodes=5,
                                                                           render=render_m, deterministic=True,
                                                                           return_episode_rewards=True)
    res = {}

    if type(var_name) == list:
        for var in var_name:
            res[var] = []
            if var in all_obs[0][0].keys():
                for ep in all_obs:
                    res[var].append([obs[var].squeeze() for obs in ep])
            elif var in all_infos[0][0].keys():
                for ep in all_infos:
                    res[var].append([info[var].squeeze() for info in ep])
            else:
                raise ValueError(f'Variable names not found in observations or infos.')
    else:
        res[var_name] = []
        if var_name in all_obs[0][0].keys():
            for ep in all_obs:
                res[var_name].append([obs[var_name].squeeze() for obs in ep])
        elif var_name in all_infos[0][0].keys():
            for ep in all_infos:
                res[var_name].append([info[var_name].squeeze() for info in ep])
        else:
            raise ValueError(f'Variable name {var_name} not found in observations or infos.')

    return res


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

run_path = "./checkpoints/StaticWaypointEnv/SingleWaypointNavigation/LOSAngleObs-Adjusted-AngVel"
model_path = run_path + "/best_model"
eval_file_path = run_path + "/evaluations.npz"
env_id = "SingleWaypointQuadXEnv-v0"

deterministic = True
render = None  # #None
render_m = False
num_eval_eps = 20

print("---------------------------- evaluate_policy --------------------------------")

# vec_env = make_vec_env(env_id=env_id, n_envs=1, seed=42)
vec_env = gym.make(env_id, render_mode=render)
# print(f'Wrapped in Monitor: {is_vecenv_wrapped(vec_env, VecMonitor) or vec_env.env_is_wrapped(Monitor)[0]}')
observations = vec_env.reset()

model = PPO.load(model_path, deterministic=deterministic)
result = aggregate_eval(model, vec_env, n_eval_episodes=num_eval_eps, var_name=['aux_state'])
smoothness = []
for ep in result["aux_state"]:
    smooth_eps = [np.linalg.norm(i) for i in ep]
    smoothness.append(smooth_eps)

# print(f'Smoothness: {smoothness}')
print(f'Auxiliary state: {result["aux_state"]}')
tmp = {'smoothness': smoothness}

plot_eval(tmp, 'smoothness')


'''
episode_rewards, episode_lengths, all_obs, all_infos = evaluate_policy(model, vec_env, n_eval_episodes=num_eval_eps, render=render_m, deterministic=deterministic, return_episode_rewards=True)
print(f'Amount of information: {len(all_obs)}')
print(len(all_obs[0]))
print(f'Information: {all_obs}')
mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)

print(f'Rewards: {episode_rewards}')
print(f'Episode lengths: {episode_lengths}')
print(f'Mean reward: {mean_reward}, Standard deviation of reward: {std_reward}')
print(f'Mean episode length: {mean_ep_length}, Standard deviation of episode length: {std_ep_length}')

print("-----------------------------------------------------------------------------")
'''
#  ---------------------------------------------------------------------------------------------------------------------
