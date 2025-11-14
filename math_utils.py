import numpy as np

# Helper methods

def skew_symmetric(v):
    """Convert a 3D vector into a 3x3 skew-symmetric matrix"""
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def unskew_symmetric(M):
    """Convert a 3x3 skew-symmetric matrix into a 3D vector"""
    return np.array([M[2,1], M[0,2], M[1,0]])

def hat_twist(xi):
    """Convert a 6D twist vector into a 4x4 matrix in se(3)"""
    v = xi[0:3]
    w = xi[3:6]
    w_skew = skew_symmetric(w)
    xi_hat = np.block([[w_skew, v.reshape((3,1))],
                       [0, 0, 0, 0]])
    return xi_hat

def unhat_twist(xi_hat):
    w_skew = xi_hat[0:3, 0:3]
    v = xi_hat[0:3, 3]
    w = unskew_symmetric(w_skew)
    return np.concatenate((v, w)).reshape((6,1))

def get_rotation(g):
    """Extract the rotation matrix from a transformation matrix T"""
    return g[0:3, 0:3]

def get_translation(g):
    """Extract the translation vector from a transformation matrix T"""
    return g[0:3, 3]

def transformation_adjoint(g):
    """Compute the adjoint representation of a transformation matrix T"""
    R = get_rotation(g)
    p = get_translation(g).reshape((3,1))
    p_skew = skew_symmetric(p.flatten())
    Ad_T = np.block([[R, p_skew @ R],
                     [np.zeros((3,3)), R]])
    return Ad_T

def joint_angles_to_rad(joint_angles_deg):
    """Convert a dictionary of joint angles in degrees to radians"""
    joint_angles_rad = {}
    for joint in joint_angles_deg:
        joint_angles_rad[joint] = np.deg2rad(joint_angles_deg[joint])
    return joint_angles_rad