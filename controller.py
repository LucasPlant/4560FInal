import so101
import time
import numpy as np
import scipy as sp
import math_utils
import noodle

np.set_printoptions(precision=4, suppress=True)

loop_freq = 15.0  # Hz
period = 1.0 / loop_freq
# timeconstant = 10
# kp = 1.0 / (loop_freq * timeconstant)  # proportional gain
kp = 0.1  # proportional gain

arm = so101.SO101()

# g_w_t = np.array([
#     [0, 1, 0, 0.1],
#     [0, 0, -1, 0.1],
#     [-1, 0, 0, 0.03],
#     [0, 0, 0, 1]
# ])

# def get_g_body_target():
#     """current filler to get the target replace with vision-based target acquisition"""
#     current_positions = arm.read_current_position()
#     g_w_e = arm.forward_kinematics(current_positions)
#     g_e_t = np.linalg.inv(g_w_e) @ g_w_t
#     # print("g_e_t:", g_e_t)
#     return g_e_t

# arm.set_position(arm.joint_array_to_dict([30, 0, 0, 0, 0, 40], gripper_value=20.0))

while True:
    start_time = time.time()

    # Read current joint positions
    current_positions_dict = arm.read_current_position()
    current_positions = np.deg2rad(arm.joint_dict_to_array(current_positions_dict))

    d_e_t = noodle.getPos_singleFrame()

    if d_e_t is None:
        print("No marker detected, holding position")
        # arm.set_position(arm.joint_array_to_dict(np.rad2deg(current_positions), gripper_value=20.0))
        time.sleep(period)
        continue

    # tangent_error = sp.linalg.logm(g_e_target)
    # xi = math_utils.unhat_twist(tangent_error)
    # print(f"Error twist: {xi}")

    error = d_e_t

    Jb = arm.body_jacobian(current_positions_dict)
    Jb_translation = Jb[0:3, :]  # Extract translational part
    Jb_pinv = np.linalg.pinv(Jb_translation)

    theta_p1_rad = current_positions + (kp * Jb_pinv @ error).flatten()

    theta_p1_deg = np.rad2deg(theta_p1_rad)

    print(f"error {error}")
    print("current position", current_positions_dict)
    print("new positions", arm.joint_array_to_dict(theta_p1_deg, gripper_value=20.0))
    print("step", np.rad2deg(theta_p1_rad) + np.rad2deg(current_positions))
    print("-----")
    print()

    arm.set_position(arm.joint_array_to_dict(theta_p1_deg, gripper_value=20.0))

    # Wait for next iteration
    end_time = time.time()
    elapsed = end_time - start_time
    # print(f"Loop time: {elapsed:.4f} seconds")
    if elapsed < period:
        time.sleep(period - elapsed)
    else:
        print(f"Warning: Control loop overran desired period {elapsed:.4f} seconds")