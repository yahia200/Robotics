import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/yahia/Documents/GitHub/robotics/install/mujoco_arm_demo'
