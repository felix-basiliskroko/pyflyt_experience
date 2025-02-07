import argparse
from stable_baselines3 import PPO, SAC, DDPG
import gymnasium as gym
import numpy as np
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import Envs.register


def load_model(algorithm, control, env):
    model_path = f"../models/{control}_control/{algorithm.lower()}_{control}_best"
    algorithms = {"PPO": PPO, "SAC": SAC, "DDPG": DDPG}
    if algorithm not in algorithms:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Choose from PPO, SAC, DDPG.")
    model = algorithms[algorithm].load(model_path)
    model.set_env(env)
    return model


def run_inference(algorithm, control, render):
    env_id = "SingleWaypointQuadXEnv-v0"
    env = gym.make(env_id, render_mode="human" if render else None, flight_mode=1 if control == "angular" else -1, reward_shift=0.75, az_align=False if render else True)
    agent_hz = env.agent_hz
    model = load_model(algorithm, control, env)
    obs, info = env.reset()
    term, trunc = False, False
    ep_reward = 0
    step_cnt = 0
    agent_positions = []

    while not (term or trunc):
        step_cnt += 1
        action, _ = model.predict(obs, deterministic=True)
        action = action.squeeze()
        obs, rew, term, trunc, info = env.step(action)
        agent_positions.append(info.get("position", [0, 0, 0]))

        if render:
            print(f'Weighted LOS reward: {info["reward_components"]["w_los_reward"]}')
            print(f'Weighted smooth reward: {info["reward_components"]["w_los_smooth_reward"]}')
            print(f'Reward: {rew}')
        else:
            pos = info.get("position", [0, 0, 0])
            distance = info.get("distance_to_target", 0)
            print(f'Position: {np.round(info["linear_position"], 3)}, Distance to Target: {round(info["distance_to_target"], 4)}, Time: {step_cnt/agent_hz}s')

            if (term or trunc):
                if info["env_complete"]:
                    print(f"--- Episode terminated: SUCCESS")
                elif info["collision"]:
                    print(f"--- Episode terminated: COLLISION")
                elif info["out_of_bounds"]:
                    print(f"--- Episode terminated: OUT OF BOUNDS")
                elif info["unstable"]:
                    print(f"--- Episode terminated: UNSTABLE")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a pretrained model.")
    parser.add_argument("--algorithm", type=str, required=True, choices=["PPO", "SAC", "DDPG"], help="RL algorithm to use.")
    parser.add_argument("--control", type=str, required=True, choices=["angular", "thrust"], help="Control type: angular or thrust.")
    parser.add_argument("--render", action="store_true", help="Enable rendering.")
    args = parser.parse_args()

    run_inference(args.algorithm, args.control, args.render)
