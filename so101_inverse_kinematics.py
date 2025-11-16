import numpy as np
from so101_forward_kinematics import Rx, Ry, Rz, get_g45, get_g5t, get_gw1, get_g12, get_g23, get_g34
def get_wrist_flex_position(target_position):
    wrist_flex_position = target_position
    wrist_flex_position[2] += .0611 + .1034
    return wrist_flex_position, None
def get_inverse_kinematics(target_position):
    x_desired, y_desired, z_desired = target_position
    r = np.sqrt(y_desired**2 + (x_desired - 0.038835)**2) - 0.0303992 # <-- Swap x and y
    z = z_desired + (0.0611 + 0.1034) - (0.0624 + 0.0542)
    
    l1 = 0.11257
    l2 = 0.028
    l3 = np.sqrt(l1**2 + l2**2)
    l4 = 0.1349
    l5 = np.sqrt(r**2 + z**2)
    
    alpha1 = np.rad2deg(np.arctan2(l2, l1))
    alpha2 = np.rad2deg(np.arccos((l3**2 + l5**2 - l4**2) / (2 * l3 * l5)))
    alpha3 = np.rad2deg(np.arctan2(z, r))
    alpha4 = np.rad2deg(np.arccos((l3**2 + l4**2 - l5**2) / (2 * l3 * l4)))
    
    theta1 = -np.rad2deg(np.arctan2(y_desired, x_desired - 0.0388353))  # <-- Swap x and y
    theta2 = 90 - alpha1 - alpha2 - alpha3
    theta3 = 90 - alpha4 + alpha1
    theta4 = 90 - theta2 - theta3
    
    joint_config = {
        'shoulder_pan': theta1,
        'shoulder_lift': theta2,
        'elbow_flex': theta3,
        'wrist_flex': theta4,
        'wrist_roll': theta1,
        'gripper': 0.0
    }
    
    return joint_config