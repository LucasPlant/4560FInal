import so101
import time
import numpy as np
import scipy as sp
import math_utils
import noodle

np.set_printoptions(precision=4, suppress=True)

loop_freq = 25  # Hz
period = 1.0 / loop_freq
# timeconstant = 10
# kp = 1.0 / (loop_freq * timeconstant)  # proportional gain
kp = 0.2  # proportional gain
# ki = 0.001
# error_i = np.zeros((3, 1))

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

counter = 0
current_positions = None
delta_theta = None
theta_p1_rad = None
current_positions_dict = arm.read_current_position()
current_positions = np.deg2rad(arm.joint_dict_to_array(current_positions_dict))

while True:
    start_time = time.time()

    if counter % 5 == 0:
        d_e_t = noodle.getPos_singleFrame()

        if d_e_t is None:
            print("No marker detected, holding position")
            # arm.set_position(arm.joint_array_to_dict(np.rad2deg(current_positions), gripper_value=20.0))
            delta_theta = np.zeros_like(current_positions)
        else:
            # compute the rotation needed to keep the tag in view
            azimuth_error = np.arctan2(d_e_t[2], d_e_t[0])
            elevation_error = np.arctan2(d_e_t[1], d_e_t[0])
            distance_error = d_e_t[0] - 0.3  # desired distance is 0.3m
            xi = np.zeros((6, 1))
            xi[4] = elevation_error  # rotation around z
            xi[5] = azimuth_error     # rotation around y
            # xi[0] = distance_error * 0.1    # translation along x


            # tangent_error = sp.linalg.logm(g_e_t)
            # xi = math_utils.unhat_twist(tangent_error)
            # Yaw around y is hard to control, so zero it out
            print(f"Error twist: {xi}")

            Jb = arm._body_jacobian(current_positions)
            # Jb_translation = Jb[0:3, :]  # Extract translational part
            #damped pseudo-inverse for numerical stability
            Jb_pinv = np.linalg.pinv(Jb, rcond=1e-3)

            delta_theta = (kp * Jb_pinv @ xi).flatten()

            print(f"error {xi}")
            print("current position", current_positions_dict)
            print("delta theta", delta_theta)
            print("-----")
            print()

    current_positions += delta_theta
    arm.set_position(arm.joint_array_to_dict(np.rad2deg(current_positions), gripper_value=20.0))

    # Wait for next iteration
    counter += 1
    end_time = time.time()
    elapsed = end_time - start_time
    # print(f"Loop time: {elapsed:.4f} seconds")
    if elapsed < period:
        time.sleep(period - elapsed)
    else:
        print(f"Warning: Control loop overran desired period {elapsed:.4f} seconds")