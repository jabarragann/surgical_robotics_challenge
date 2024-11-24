from .ral_abstract import RAL_Abstract
import rclpy
from rclpy.node import Node


class RAL_ROS2(RAL_Abstract):

    def __init__(self, node: Node):
        self.node = node

    def identify(self) -> str:
        return "ROS2"

    def is_shutdown(self):
        return not rclpy.ok()

    def shutdown(self):
        rclpy.shutdown()

    def create_rate(self, rate_hz):
        return self.node.create_rate(rate_hz)

    def now(self):
        clock = self.node.get_clock()
        return clock.now()
