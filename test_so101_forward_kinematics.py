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
    random_configurations.append({
        'shoulder_pan': np.random.uniform(-90, 90),
        'shoulder_lift': np.random.uniform(-90, 90),
        'elbow_flex': np.random.uniform(-90, 90),
        'wrist_flex': np.random.uniform(-90, 90),
        'wrist_roll': np.random.uniform(-90, 90),
        'gripper': 0.0
    })

set_initial_pose(d, random_configurations[0])
send_position_command(d, random_configurations[0])

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  # add cubes to show start and end positions
  show_cubes(viewer, random_configurations)

  for i in range(num_random_configs):
    move_to_pose(m, d, viewer, random_configurations[i], 5.0)
    # Hold Starting Position for 4 seconds
    hold_position(m, d, viewer, 4.0)
