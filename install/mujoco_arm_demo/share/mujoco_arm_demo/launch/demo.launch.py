from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package="mujoco_arm_demo",
            executable="simple_move_node",
            name="mujoco_arm_node",
            output="screen",
        )
    ])
