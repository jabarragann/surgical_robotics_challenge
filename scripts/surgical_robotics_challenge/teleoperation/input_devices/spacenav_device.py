import rospy
from PyKDL import Frame, Vector, Rotation
from sensor_msgs.msg import Joy
from ambf_client import Client
import sys
import time
from argparse import ArgumentParser
import numpy as np


class SpaceNavDevice:
    def __init__(self, name="/spacenav/"):

        if not rospy.get_node_uri():
            rospy.init_node("image_saver_node", anonymous=True, log_level=rospy.WARN)
        else:
            rospy.logdebug(rospy.get_caller_id() + " -> ROS already initialized")

        joy_str = name + "joy"
        self.__ecm_joints_velocity = np.zeros(4)

        self._pose_sub = rospy.Subscriber(joy_str, Joy, self.joy_cb, queue_size=10)

        time.sleep(0.5)

    def joy_cb(self, msg):
        """Spacenav joystick msg description. Refer to documentation for more details.
        msg.axes[0] = zoom motion
        msg.axes[1] = pan left/right
        msg.axes[2] = pan up/down
        msg.axes[3] = roll
        msg.axes[4] = tilt
        msg.axes[5] = spin

        """
        j1_vel = msg.axes[3]  # roll
        j2_vel = msg.axes[4]  # tilt
        j3_vel = msg.axes[0]  # zoom motion
        j4_vel = msg.axes[5]  # spin

        self.__ecm_joints_velocity[:] = [j1_vel, j2_vel, j3_vel, j4_vel]

    def get_joints_velocity(self):
        return self.__ecm_joints_velocity


def main():
    parser = ArgumentParser()

    # fmt: off
    parser.add_argument( "-d", action="store", dest="spacenav_name", 
                        help="Specify ros base name of spacenav", default="/spacenav/")
    # fmt: on

    parsed_args = parser.parse_args()
    print(f"Specified Arguments: {parsed_args}")

    _spacenav_one_name = parsed_args.spacenav_name

    spacenav_one = SpaceNavDevice(_spacenav_one_name)

    _pub_freq = 500
    rate = rospy.Rate(_pub_freq)

    while not rospy.is_shutdown():
        velocity = spacenav_one.get_joints_velocity()
        print(
            f"ECM Joints Velocity J1:{velocity[0]: 0.3f} J2:{velocity[1]: 0.3f}"
            + f" J3:{velocity[2]: 0.3f} J4:{velocity[3]: 0.3f}",
            end="\r",
        )
        rospy.sleep(0.01)


if __name__ == "__main__":
    main()
