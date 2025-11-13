# File full of utils for interacting with the SO101 robot
import numpy as np
import scipy as sp

# Helper methods
def hat_twist(xi):
    """Convert a 6D twist vector into a 4x4 matrix in se(3)"""
    v = xi[0:3].flatten()
    w = xi[3:6].flatten()
    w_skew = np.array([[0, -w[2], w[1]],
                      [w[2], 0, -w[0]],
                      [-w[1], w[0], 0]])
    xi_hat = np.block([[w_skew, v.reshape((3,1))],
                       [0, 0, 0, 0]])
    return xi_hat

class SO101:
    
    # Params


    # Forward Kinematic Parameters
    x_offsets = [0.0388353, 0.0303992, 0.028, 0.1349, 0.0611, 0.1034]
    y_offsets = [0.0]
    z_offsets = [0.0624, 0.0542, 0.11257]
    g0 = np.array([[ 1,  0,  0, sum(x_offsets)],
                   [  0,  1,  0, sum(y_offsets)],
                   [  0,  1,  -1, sum(z_offsets)],
                   [  0,  0,  0, 1.0]])
    
    # maps the joint to a tuple of (w, q) for the twist
    # Gripper not included here
    joint_twist_info = {
        'shoulder_pan':    (np.array([[0], [0], [-1]]), np.array([[x_offsets[0]], [0], [z_offsets[0]]])),
        'shoulder_lift':   (np.array([[0], [1], [0]]), np.array([[sum(x_offsets[:2])], [0.0], [sum(z_offsets[:2])]])),
        'elbow_flex':      (np.array([[0], [1], [0]]), np.array([[sum(x_offsets[:3])], [0.0], [sum(z_offsets)]])),
        'wrist_flex':      (np.array([[0], [1], [0]]), np.array([[sum(x_offsets[:4])], [0.0], [sum(z_offsets)]])),
        'wrist_roll':      (np.array([[-1], [0], [0]]), np.array([[sum(x_offsets[:5])], [0.0], [sum(z_offsets)]])),
    }

    # Compute the spacial joint twists
    joint_twists = {}
    for joint in joint_twist_info:
        w, q = joint_twist_info[joint]
        print("joint:", joint, " w:", w.flatten(), " q:", q.flatten())
        v = -np.cross(w.flatten(), q.flatten()).reshape((3,1))
        joint_twists[joint] = np.vstack((v, w))

    @classmethod
    def _forward_kinematics(cls, joint_angles):
        """
        Compute the forward kinematics for the SO101 robot given joint angles.
        Angles in Radians
        """
        wge = np.eye(4)
        for joint in cls.joint_twists:
            w = cls.joint_twists[joint]
            theta = joint_angles[joint]
            wge = wge @ sp.linalg.expm(hat_twist(w) * theta)
        wge = wge @ cls.g0
        return wge
    
    @classmethod
    def forward_kinematics(cls, joint_angles):
        """
        Compute the forward kinematics for the SO101 robot given joint angles.
        Angles in DEGREES
        """
        # Convert to radians
        joint_angles_rad = {}
        for joint in joint_angles:
            joint_angles_rad[joint] = np.deg2rad(joint_angles[joint])
        return cls._forward_kinematics(joint_angles_rad)
