#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import math
import time

class ArmMover(Node):
    def __init__(self):
        super().__init__('arm_mover')
        self.publisher = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.start_time = time.time()
        self.timer = self.create_timer(0.02, self.publish_command)  # 50 Hz

    def publish_command(self):
        t = time.time() - self.start_time
        msg = Float64MultiArray()
        # adjust number of joints depending on your model
        # print(t/4)
        msg.data = [
            math.pi/2,
            0.2 * math.cos(2*t),
            0.0,
            0.0,
            -0.1 * math.sin(t),
            2 * math.sin(2*t),
        ]
        self.publisher.publish(msg)

def main():
    rclpy.init()
    node = ArmMover()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
