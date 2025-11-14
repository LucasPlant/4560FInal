import time
import mujoco
import mujoco.viewer
from so101_mujoco_utils import set_initial_pose, send_position_command, move_to_pose, hold_position
from so101 import SO101
import numpy as np

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

def show_cubes(viewer, configs, halfwidth=0.013):
    for config in configs:
        wge = SO101.forward_kinematics(config)
        object_position = wge[0:3, 3]
        object_orientation = wge[0:3, 0:3]
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_BOX, 
            size=[halfwidth, halfwidth, halfwidth],                 
            pos=object_position,                         
            mat=object_orientation.flatten(),              
            rgba=[1, 0, 0, 0.2]                           
        )
        viewer.user_scn.ngeom += 1
    viewer.sync()
    return

test_configuration = {
    'shoulder_pan': 30,   # in radians for mujoco! 
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0
}

num_random_configs = 5
random_configurations = []

for i in range(num_random_configs):
    # Sample each joint uniformly within the limits defined in SO101.joint_limits.
    # SO101.joint_limits are in radians; convert sampled values to degrees because
    # the test code (and SO101.forward_kinematics) expect joint values in degrees.
    cfg = {}
    for joint in SO101.ordered_joints:
        low_rad, high_rad = SO101.joint_limits[joint]
        # sample in radians then convert to degrees
        cfg[joint] = np.rad2deg(np.random.uniform(low_rad, high_rad))
    # gripper limit is stored under 'gripper' in radians as well
    g_low, g_high = SO101.joint_limits.get('gripper', (0.0, 0.0))
    cfg['gripper'] = np.rad2deg(np.random.uniform(g_low, g_high))
    random_configurations.append(cfg)

end_effector_targets = [SO101.forward_kinematics(config) for config in random_configurations]
computed_configurations = [SO101.inverse_kinematics(target) for target in end_effector_targets]

set_initial_pose(d, random_configurations[0])
send_position_command(d, random_configurations[0])

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  # add cubes to show start and end positions
  show_cubes(viewer, random_configurations)

  for i in range(num_random_configs):
      move_to_pose(m, d, viewer, computed_configurations[i], 5.0)
      # Hold Starting Position for 4 seconds
      hold_position(m, d, viewer, 4.0)
