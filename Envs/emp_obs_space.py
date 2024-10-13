import random

import gymnasium
import numpy as np
import PyFlyt.gym_envs
from tqdm import tqdm

import register  # needed to register the custom environments in gymnasium
from WaypointEnv import QuadXWaypoint

# env = gymnasium.make("Quadx-Waypoint-v0", render_mode="human")
env = gymnasium.make("Quadx-Waypoint-v0", render_mode=None)

# Attitude ranges
ang_vel_min, ang_vel_max = 0, 0
ang_pos_min, ang_pos_max = 0, 0
lin_vel_min, lin_vel_max = 0, 0
lin_pos_min, lin_pos_max = 0, 0

lin_pos_x_mean, lin_pos_y_mean, lin_pos_z_mean = [], [], []
lin_vel_x_mean, lin_vel_y_mean, lin_vel_z_mean = [], [], []
prev_dist_mean = []

quat_min, quat_max = 0, 0

# Action ranges
thrust_min, thrust_max = 0, 0

# Auxiliary ranges
aux_min, aux_max = 0, 0


term, trunc = False, False
obs, _ = env.reset()
save_interval = 2_000

for i in tqdm(range(2_000)):
    while not (term or trunc):
        new_action = env.action_space.sample()
        obs, rew, term, trunc, _ = env.step(new_action)
        ang_vel_x, ang_vel_y, ang_vel_z = obs["attitude"][0][0:3]
        ang_pos_x, ang_pos_y, ang_pos_z = obs["attitude"][0][3:6]
        lin_vel_x, lin_vel_y, lin_vel_z = obs["attitude"][0][6:9]
        lin_pos_x, lin_pos_y, lin_pos_z = obs["attitude"][0][9:12]
        quat = obs["attitude"][0][12:16]
        prev_action_1, prev_action_2, prev_action_3, prev_action_4 = obs["prev_action"][0][0], obs["prev_action"][0][1], obs["prev_action"][0][2], obs["prev_action"][0][3]
        aux_1, aux_2, aux_3, aux_4 = obs["auxiliary"][0][0], obs["auxiliary"][0][1], obs["auxiliary"][0][2], obs["auxiliary"][0][3]
        target_delta_x, target_delta_y, target_delta_z = obs["target_delta"][0][0], obs["target_delta"][0][1], obs["target_delta"][0][2]
        prev_dist = obs["previous_dist"][0]

        if ang_vel_x < ang_vel_min:
            ang_vel_min = ang_vel_x
        if ang_vel_x > ang_vel_max:
            ang_vel_max = ang_vel_x

        if ang_vel_y < ang_vel_min:
            ang_vel_min = ang_vel_y
        if ang_vel_y > ang_vel_max:
            ang_vel_max = ang_vel_y

        if ang_vel_z < ang_vel_min:
            ang_vel_min = ang_vel_z
        if ang_vel_z > ang_vel_max:
            ang_vel_max = ang_vel_z

        if ang_pos_x < ang_pos_min:
            ang_pos_min = ang_pos_x
        if ang_pos_x > ang_pos_max:
            ang_pos_max = ang_pos_x

        if ang_pos_y < ang_pos_min:
            ang_pos_min = ang_pos_y
        if ang_pos_y > ang_pos_max:
            ang_pos_max = ang_pos_y

        if ang_pos_z < ang_pos_min:
            ang_pos_min = ang_pos_z
        if ang_pos_z > ang_pos_max:
            ang_pos_max = ang_pos_z

        if lin_vel_x < lin_vel_min:
            lin_vel_min = lin_vel_x
        if lin_vel_x > lin_vel_max:
            lin_vel_max = lin_vel_x

        if lin_vel_y < lin_vel_min:
            lin_vel_min = lin_vel_y
        if lin_vel_y > lin_vel_max:
            lin_vel_max = lin_vel_y

        if lin_vel_z < lin_vel_min:
            lin_vel_min = lin_vel_z
        if lin_vel_z > lin_vel_max:
            lin_vel_max = lin_vel_z

        if lin_pos_x < lin_pos_min:
            lin_pos_min = lin_pos_x
        if lin_pos_x > lin_pos_max:
            lin_pos_max = lin_pos_x

        if lin_pos_y < lin_pos_min:
            lin_pos_min = lin_pos_y
        if lin_pos_y > lin_pos_max:
            lin_pos_max = lin_pos_y

        if lin_pos_z < lin_pos_min:
            lin_pos_min = lin_pos_z
        if lin_pos_z > lin_pos_max:
            lin_pos_max = lin_pos_z

        # Quaternions are between -1 and 1
        if quat[0] < quat_min:
            quat_min = quat[0]
        if quat[0] > quat_max:
            quat_max = quat[0]

        if quat[1] < quat_min:
            quat_min = quat[1]
        if quat[1] > quat_max:
            quat_max = quat[1]

        if quat[2] < quat_min:
            quat_min = quat[2]
        if quat[2] > quat_max:
            quat_max = quat[2]

        if quat[3] < quat_min:
            quat_min = quat[3]
        if quat[3] > quat_max:
            quat_max = quat[3]

        # Action values
        if prev_action_1 < thrust_min:
            thrust_min = prev_action_1
        if prev_action_1 > thrust_max:
            thrust_max = prev_action_1

        if prev_action_2 < thrust_min:
            thrust_min = prev_action_2
        if prev_action_2 > thrust_max:
            thrust_max = prev_action_2

        if prev_action_3 < thrust_min:
            thrust_min = prev_action_3
        if prev_action_3 > thrust_max:
            thrust_max = prev_action_3

        if prev_action_4 < thrust_min:
            thrust_min = prev_action_4
        if prev_action_4 > thrust_max:
            thrust_max = prev_action_4

        # Auxiliary values
        if aux_1 < aux_min:
            aux_min = aux_1
        if aux_1 > aux_max:
            aux_max = aux_1

        if aux_2 < aux_min:
            aux_min = aux_2
        if aux_2 > aux_max:
            aux_max = aux_2

        if aux_3 < aux_min:
            aux_min = aux_3
        if aux_3 > aux_max:
            aux_max = aux_3

        if aux_4 < aux_min:
            aux_min = aux_4
        if aux_4 > aux_max:
            aux_max = aux_4

        # Calculate moving average:
        lin_pos_x_mean.append(lin_pos_x)
        lin_pos_y_mean.append(lin_pos_y)
        lin_pos_z_mean.append(lin_pos_z)

        lin_vel_x_mean.append(lin_vel_x)
        lin_vel_y_mean.append(lin_vel_y)
        lin_vel_z_mean.append(lin_vel_z)

        prev_dist_mean.append(prev_dist)

        if i % save_interval == -1:
            data = {
                "Angular Velocity": {
                    "X": f"{ang_vel_min} - {ang_vel_max}",
                    "Y": f"{ang_vel_min} - {ang_vel_max}",
                    "Z": f"{ang_vel_min} - {ang_vel_max}"
                },
                "Angular Position": {
                    "X": f"{ang_pos_min} - {ang_pos_max}",
                    "Y": f"{ang_pos_min} - {ang_pos_max}",
                    "Z": f"{ang_pos_min} - {ang_pos_max}"
                },
                "Linear Velocity": {
                    "X": f"{lin_vel_min} - {lin_vel_max}",
                    "Y": f"{lin_vel_min} - {lin_vel_max}",
                    "Z": f"{lin_vel_min} - {lin_vel_max}"
                },
                "Linear Position": {
                    "X": f"{lin_pos_min} - {lin_pos_max}",
                    "Y": f"{lin_pos_min} - {lin_pos_max}",
                    "Z": f"{lin_pos_min} - {lin_pos_max}"
                },
                "Quaternions": f"{quat_min} - {quat_max}",
                "Action (Thrust)": f"{thrust_min} - {thrust_max}",
                "Auxiliary": f"{aux_min} - {aux_max}",
                "Averaged Linear Velocity": {
                    "X": {
                        "Mean": np.mean(lin_vel_x_mean),
                        "Standard Deviation": np.std(lin_vel_x_mean)
                    },
                    "Y": {
                        "Mean": np.mean(lin_vel_y_mean),
                        "Standard Deviation": np.std(lin_vel_y_mean)
                    },
                    "Z": {
                        "Mean": np.mean(lin_vel_z_mean),
                        "Standard Deviation": np.std(lin_vel_z_mean)
                    }
                },
                "Averaged Linear Position": {
                    "X": {
                        "Mean": np.mean(lin_pos_x_mean),
                        "Standard Deviation": np.std(lin_pos_x_mean)
                    },
                    "Y": {
                        "Mean": np.mean(lin_pos_y_mean),
                        "Standard Deviation": np.std(lin_pos_y_mean)
                    },
                    "Z": {
                        "Mean": np.mean(lin_pos_z_mean),
                        "Standard Deviation": np.std(lin_pos_z_mean)
                    }
                },
                "Averaged Previous Distance": {
                    "Mean": np.mean(prev_dist_mean),
                    "Standard Deviation": np.std(prev_dist_mean)
                }
            }

            with open("ema_parameters_2.json", "w") as f:
                f.write(str(data))

    obs, _ = env.reset()

print(f'Angular Velocity (X): {ang_vel_min} - {ang_vel_max}')
print(f'Angular Velocity (Y): {ang_vel_min} - {ang_vel_max}')
print(f'Angular Velocity (Z): {ang_vel_min} - {ang_vel_max}')

print(f'Angular Position (X): {ang_pos_min} - {ang_pos_max}')
print(f'Angular Position (Y): {ang_pos_min} - {ang_pos_max}')
print(f'Angular Position (Z): {ang_pos_min} - {ang_pos_max}')

print(f'Linear Velocity (X): {lin_vel_min} - {lin_vel_max}')
print(f'Linear Velocity (Y): {lin_vel_min} - {lin_vel_max}')
print(f'Linear Velocity (Z): {lin_vel_min} - {lin_vel_max}')

print(f'Linear Position (X): {lin_pos_min} - {lin_pos_max}')
print(f'Linear Position (Y): {lin_pos_min} - {lin_pos_max}')
print(f'Linear Position (Z): {lin_pos_min} - {lin_pos_max}')

print("-------------------------------------------------------------------")

print(f'Quaternions: {quat_min} - {quat_max}')
print(f'Action (Thrust): {thrust_min} - {thrust_max}')

print(f'Auxiliary: {aux_min} - {aux_max}')

print("-------------------------------------------------------------------")

print(f'Averaged Linear Velocity (X): {np.mean(lin_vel_x_mean)}')
print(f'Standard Deviation of Linear Velocity (X): {np.std(lin_vel_x_mean)}')

print(f'Averaged Linear Velocity (Y): {np.mean(lin_vel_y_mean)}')
print(f'Standard Deviation of Linear Velocity (Y): {np.std(lin_vel_y_mean)}')

print(f'Averaged Linear Velocity (Z): {np.mean(lin_vel_z_mean)}')
print(f'Standard Deviation of Linear Velocity (Z): {np.std(lin_vel_z_mean)}')

print("-------------------------------------------------------------------")

print(f'Averaged Linear Position (X): {np.mean(lin_pos_x_mean)}')
print(f'Standard Deviation of Linear Position (X): {np.std(lin_pos_x_mean)}')

print(f'Averaged Linear Position (Y): {np.mean(lin_pos_y_mean)}')
print(f'Standard Deviation of Linear Position (Y): {np.std(lin_pos_y_mean)}')

print(f'Averaged Linear Position (Z): {np.mean(lin_pos_z_mean)}')
print(f'Standard Deviation of Linear Position (Z): {np.std(lin_pos_z_mean)}')

print("-------------------------------------------------------------------")

print(f'Averaged Previous Distance: {np.mean(prev_dist_mean)}')
print(f'Standard Deviation of Previous Distance: {np.std(prev_dist_mean)}')



