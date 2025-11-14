# File full of utils for interacting with the SO101 robot
import numpy as np
import scipy as sp
from math_utils import hat_twist, unhat_twist, joint_angles_to_rad, transformation_adjoint

class SO101:
    
    # Params
    ordered_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    num_joints = len(ordered_joints)

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
        'shoulder_pan':    (np.array([0, 0, -1]), np.array([x_offsets[0], 0, z_offsets[0]])),
        'shoulder_lift':   (np.array([0, 1, 0]), np.array([sum(x_offsets[:2]), 0.0, sum(z_offsets[:2])])),
        'elbow_flex':      (np.array([0, 1, 0]), np.array([sum(x_offsets[:3]), 0.0, sum(z_offsets)])),
        'wrist_flex':      (np.array([0, 1, 0]), np.array([sum(x_offsets[:4]), 0.0, sum(z_offsets)])),
        'wrist_roll':      (np.array([-1, 0, 0]), np.array([sum(x_offsets[:5]), 0.0, sum(z_offsets)])),
    }

    # Compute the spacial joint twists
    joint_twists = []
    for joint in ordered_joints:
        w, q = joint_twist_info[joint]
        v = -np.cross(w, q)
        joint_twists.append(np.concatenate((v, w)))

    @classmethod
    def _forward_kinematics(cls, joint_angles):
        """
        Compute the forward kinematics for the SO101 robot given joint angles.
        Angles in Radians
        """
        wge = np.eye(4)
        for i, joint in enumerate(cls.ordered_joints):
            xi = cls.joint_twists[i]
            theta = joint_angles[joint]
            wge = wge @ sp.linalg.expm(hat_twist(xi) * theta)
        wge = wge @ cls.g0
        return wge
    
    @classmethod
    def forward_kinematics(cls, joint_angles):
        """
        Compute the forward kinematics for the SO101 robot given joint angles.
        Angles in DEGREES
        """
        # Convert to radians
        return cls._forward_kinematics(joint_angles_to_rad(joint_angles))

    @classmethod
    def _spatial_jacobian(cls, joint_angles):
        """
        Compute the spatial Jacobian for the SO101 robot given joint angles.
        Angles in Rad
        """
        Js = np.zeros((6, len(cls.joint_twists)))

        cumulative_transform = np.eye(4)

        for i, joint in enumerate(cls.ordered_joints):
            xi = cls.joint_twists[i]
            xi_prime_hat = transformation_adjoint(cumulative_transform) @ xi
            Js[:, i] = xi_prime_hat
            cumulative_transform = cumulative_transform @ sp.linalg.expm(hat_twist(xi) * joint_angles[joint])

        return Js
    
    @classmethod
    def spatial_jacobian(cls, joint_angles):
        """
        Compute the spatial Jacobian for the SO101 robot given joint angles.
        Angles in DEGREES
        """
        # Convert to radians
        return cls._spatial_jacobian(joint_angles_to_rad(joint_angles))
    
    @classmethod
    def _body_jacobian(cls, joint_angles):
        """
        Compute the body Jacobian for the SO101 robot given joint angles.
        Angles in Rad
        """
        Jb = np.zeros((6, len(cls.joint_twists)))

        cumulative_transform = cls.g0.copy()

        for i, joint in reversed(list(enumerate(cls.ordered_joints))):
            xi = cls.joint_twists[i]
            xi_dagger = transformation_adjoint(np.linalg.inv(cumulative_transform)) @ xi
            Jb[:, i] = xi_prime_hat
            cumulative_transform = sp.linalg.expm(hat_twist(xi) * joint_angles[joint]) @ cumulative_transform

        return Jb

