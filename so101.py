# File full of utils for interacting with the SO101 robot
import numpy as np
import scipy as sp
from math_utils import hat_twist, unhat_twist, transformation_adjoint, tangent_space_error, transformation_matrix
import time
from so101_utils import setup_motors, load_calibration

# CONFIGURATION VARIABLES

DEFAULT_PORT_ID = "/dev/tty.usbmodem5A7A0548111" # REPLACE WITH YOUR PORT! 
DEFAULT_ROBOT_NAME = "my_follower" # REPLACE WITH YOUR ROBOT NAME! 

class SO101:
    
    #===================================================================
    ############### Robot Kinematics ################
    #===================================================================
    # Params
    ordered_joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    num_joints = len(ordered_joints)
    joint_limits = {
        'shoulder_pan': (-1.919, 1.919),
        'shoulder_lift': (-1.74, 1.74),
        'elbow_flex': (-1.69, 1.69),
        'wrist_flex': (-1.65, 1.65),
        'wrist_roll': (-2.74, 2.84),
        'gripper': (0, np.pi/2)
    }
    theta_min = np.zeros(num_joints)
    theta_max = np.zeros(num_joints)
    for i, joint in enumerate(ordered_joints):
        theta_min[i] = joint_limits[joint][0]
        theta_max[i] = joint_limits[joint][1]

    # Forward Kinematic Parameters
    x_offsets = [0.0388353, 0.0303992, 0.028, 0.1349, 0.0611, 0.1034]
    y_offsets = [0.0]
    z_offsets = [0.0624, 0.0542, 0.11257]
    g0 = np.array([[ 1,  0,  0, sum(x_offsets)],
                   [  0,  0,  -1, sum(y_offsets)],
                   [  0,  1,  0, sum(z_offsets)],
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
    def joint_dict_to_array(cls, joint_dict):
        """
        Convert a dictionary of joint angles to a numpy array in the order defined by ordered_joints.
        """
        joint_array = np.zeros(cls.num_joints)
        for i, joint in enumerate(cls.ordered_joints):
            joint_array[i] = joint_dict[joint]
        return joint_array
    
    @classmethod
    def joint_array_to_dict(cls, joint_array, gripper_value=0.0):
        """
        Convert a numpy array of joint angles to a dictionary in the order defined by ordered_joints.
        Angles in Radians
        """
        joint_dict = {}
        for i, joint in enumerate(cls.ordered_joints):
            joint_dict[joint] = joint_array[i]
        joint_dict['gripper'] = gripper_value
        return joint_dict

    @classmethod
    def _forward_kinematics(cls, joint_angles):
        """
        Compute the forward kinematics for the SO101 robot given joint angles.
        Angles in Radians
        """
        wge = np.eye(4)
        for i in range(cls.num_joints):
            xi = cls.joint_twists[i]
            theta = joint_angles[i]
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
        joint_angles = cls.joint_dict_to_array(joint_angles)
        return cls._forward_kinematics(np.deg2rad(joint_angles))

    @classmethod
    def _spatial_jacobian(cls, joint_angles):
        """
        Compute the spatial Jacobian for the SO101 robot given joint angles.
        Angles in Rad
        """
        Js = np.zeros((6, len(cls.joint_twists)))

        cumulative_transform = np.eye(4)

        for i in range(cls.num_joints):
            xi = cls.joint_twists[i]
            xi_hat = hat_twist(xi)
            xi_prime_hat = cumulative_transform @ xi_hat @ np.linalg.inv(cumulative_transform)
            xi_prime = unhat_twist(xi_prime_hat)
            Js[:, i] = xi_prime
            cumulative_transform = cumulative_transform @ sp.linalg.expm(hat_twist(xi) * joint_angles[i])

        return Js
    
    @classmethod
    def spatial_jacobian(cls, joint_angles):
        """
        Compute the spatial Jacobian for the SO101 robot given joint angles.
        Angles in DEGREES
        """
        # Convert to radians
        joint_angles = cls.joint_dict_to_array(joint_angles)
        return cls._spatial_jacobian(np.deg2rad(joint_angles))
    
    @classmethod
    def _body_jacobian(cls, joint_angles):
        """
        Compute the body Jacobian for the SO101 robot given joint angles.
        Angles in Rad
        """
        Jb = np.zeros((6, len(cls.joint_twists)))

        cumulative_transform = cls.g0.copy()

        for i in reversed(range(cls.num_joints)):
            xi = cls.joint_twists[i]
            cumulative_transform = sp.linalg.expm(hat_twist(xi) * joint_angles[i]) @ cumulative_transform
            xi_hat = hat_twist(xi)
            xi_dagger_hat = np.linalg.inv(cumulative_transform) @ xi_hat @ cumulative_transform
            xi_dagger = unhat_twist(xi_dagger_hat)
            Jb[:, i] = xi_dagger

        return Jb
    
    @classmethod
    def body_jacobian(cls, joint_angles):
        """
        Compute the body Jacobian for the SO101 robot given joint angles.
        Angles in DEGREES
        """
        # Convert to radians
        joint_angles = cls.joint_dict_to_array(joint_angles)
        return cls._body_jacobian(np.deg2rad(joint_angles))
    
    @classmethod
    def _inverse_kinematics(cls, desired_wge, initial_joint_angles=None, tol=1e-10, max_iters=1000, max_attempts=5):
        start_time = time.time()

        def error_func(joint_angles):
            wge = cls._forward_kinematics(joint_angles)
            error = tangent_space_error(wge, desired_wge)          # shape (6,) or (6,1)

            error = np.asarray(error).reshape(-1)                  # make sure 1D
            f = float(error @ error)                               # scalar

            Jb = cls._body_jacobian(joint_angles)                  # 6×n
            grad = -2.0 * (Jb.T @ error)                            # n×1 → n,
            grad = np.asarray(grad).reshape(-1)                    # ensure 1D

            # return f
            return f, grad

        if initial_joint_angles is None:
            initial_joint_angles = np.zeros(cls.num_joints)

        # NOTE: BFGS ignores bounds. Use L-BFGS-B if you actually want these.
        # bounds = list(zip(cls.theta_min, cls.theta_max))

        result = sp.optimize.minimize(
            fun=error_func,
            x0=initial_joint_angles,
            method='BFGS',
            jac=True,                      # tell SciPy fun returns (f, grad)
            tol=tol,
            options={'maxiter': max_iters}
        )

        end_time = time.time()
        print("================= Inverse Kinematics Result ================")
        print(f"IK computation time: {end_time - start_time:.4f} seconds")
        f_final, _ = error_func(result.x)
        # f_final = error_func(result.x)
        print(f"IK optimization success: {result.success}, message: {result.message}")
        print(f"Final IK error norm: {np.sqrt(f_final)}")
        print("============================================================")

        return result.x

    
    @classmethod
    def inverse_kinematics(cls, desired_wge, initial_joint_angles=None, tol=1e-10, max_iters=1000, max_attempts=5):
        """
        Compute the inverse kinematics for the SO101 robot using Newton-Raphson method.
        Angles in DEGREES
        """
        if initial_joint_angles is not None:
            initial_joint_angles = cls.joint_dict_to_array(initial_joint_angles)
            initial_joint_angles = np.deg2rad(initial_joint_angles)
        ik_solution_rad = cls._inverse_kinematics(desired_wge, initial_joint_angles, tol, max_iters, max_attempts)
        return cls.joint_array_to_dict(np.rad2deg(ik_solution_rad))

    #===================================================================
    ############### Robot Control ################
    #===================================================================

    def __init__(self, port = DEFAULT_PORT_ID, robot_name = DEFAULT_ROBOT_NAME):
        self.calibration = load_calibration(robot_name)
        self.bus = setup_motors(self.calibration, port)

    def read_current_position(self):
        return self.bus.sync_read("Present_Position")
    
    def set_position(self, joint_positions):
        self.bus.sync_write("Goal_Position", joint_positions, normalize=True)

    def __del__(self):
        self.bus.torque_disabled()
        self.bus.disconnect()
