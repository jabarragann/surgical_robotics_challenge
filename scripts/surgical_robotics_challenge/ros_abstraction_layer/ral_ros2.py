from .ral_abstract import RAL_Abstract 
import rclpy 

# TODO - Need to pass _node handle from AMBF_CLIENT

class RAL_ROS2(RAL_Abstract):

    def identify(self)->str:
        return "ROS2"

    def is_shutdown(self):
        return not rclpy.ok()
    
    def shutdown(self):
        rclpy.shutdown()

    def create_rate(self, rate_hz):
        return self._node.create_rate(rate_hz)

    def now(self):
        clock = self._node.get_clock()
        return clock.now()
