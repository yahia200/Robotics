#!/usr/bin/env python3
import time
import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class MujocoArmNode(Node):
    def __init__(self):
        super().__init__("mujoco_arm_node")

        # Path to your Mujoco model
        model_path = "/home/YOUR_USERNAME/mujoco_menagerie/low_cost_arm/scene.xml"
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Initialize viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # Subscribe to /joint_command topic
        self.subscription = self.create_subscription(
            Float64MultiArray, "joint_command", self.command_callback, 10
        )
        self.subscription  # prevent unused variable warning

        self.get_logger().info("Mujoco arm simulation started. Listening to /joint_command")

        # Timer for stepping the simulation
        self.timer = self.create_timer(0.01, self.sim_step)

    def command_callback(self, msg):
        """Receives a list of joint control commands (torques or positions)."""
        for i, val in enumerate(msg.data):
            if i < len(self.data.ctrl):
                self.data.ctrl[i] = val

    def sim_step(self):
        mujoco.mj_step(self.model, self.data)
        self.viewer.sync()

    def destroy_node(self):
        self.viewer.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MujocoArmNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
