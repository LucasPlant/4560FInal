import so101
import time
import numpy as np
import scipy as sp
import math_utils
import noodle

np.set_printoptions(precision=4, suppress=True)

loop_freq = 25  # Hz
period = 1.0 / loop_freq
kp = 0.1  # proportional gain

arm = so101.SO101()

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
            xi = np.zeros((6, 1))
            xi[4] = elevation_error  # rotation around z
            xi[5] = azimuth_error     # rotation around y
            xi[1] = - d_e_t[1] * 5.0  # translation along y
            xi[0] = (d_e_t[0] - 0.5) * 0.1    # translation along x


            # tangent_error = sp.linalg.logm(g_e_t)
            # xi = math_utils.unhat_twist(tangent_error)
            # Yaw around y is hard to control, so zero it out
            print(f"Error twist: {xi}")

            Jb = arm._body_jacobian(current_positions)
            # Jb_translation = Jb[0:3, :]  # Extract translational part
            #damped pseudo-inverse for numerical stability
            Jb_pinv = np.linalg.pinv(Jb, rcond=1e-2)

            delta_theta = (kp * Jb_pinv @ xi).flatten()

            print(f"error {xi}")
            print("current position", current_positions_dict)
            print("delta theta", delta_theta)
            print("-----")
            print()

    current_positions += delta_theta
    current_positions_temp = current_positions.copy()
    current_positions_temp[4] = -np.pi / 2
    arm.set_position(arm.joint_array_to_dict(np.rad2deg(current_positions_temp), gripper_value=20.0))

    # Wait for next iteration
    counter += 1
    end_time = time.time()
    elapsed = end_time - start_time
    # print(f"Loop time: {elapsed:.4f} seconds")
    if elapsed < period:
        time.sleep(period - elapsed)
    else:
        print(f"Warning: Control loop overran desired period {elapsed:.4f} seconds")