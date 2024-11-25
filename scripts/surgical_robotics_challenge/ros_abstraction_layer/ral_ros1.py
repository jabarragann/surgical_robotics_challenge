from .ral_abstract import RAL_Abstract
import rospy


class RAL_ROS1(RAL_Abstract):
    def __init__(self, node):
        # Node is not used in ROS1
        pass

    def identify(self) -> str:
        return "ROS1"

    def is_shutdown(self):
        return rospy.is_shutdown()

    def shutdown(self):
        # Not need for ROS1
        pass

    def create_rate(self, rate_hz):
        return rospy.Rate(rate_hz)

    def now(self):
        return rospy.Time.now().to_sec()
