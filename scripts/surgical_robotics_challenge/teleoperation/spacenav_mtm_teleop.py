from surgical_robotics_challenge.ecm_arm import ECM
import time
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.utils.jnt_control_gui import JointGUI
import rospy
import numpy as np
from surgical_robotics_challenge.teleoperation.input_devices.spacenav_device import SpaceNavDevice
import argparse
import sys


class ControlInterface:
    def __init__(self, scale=0.25) -> None:

        simulation_manager = SimulationManager("ECM_spacenav_control")
        time.sleep(0.5)
        self.ecm = ECM(simulation_manager, "CameraFrame")
        self.spacenav = SpaceNavDevice()

        self.scale = scale
        self.joint_specific_scale = np.array([8, 8, 1.2, 8]) * self.scale
        self.dt = 0.005

    def run(self):
        total_scale = self.scale * self.joint_specific_scale
        while not rospy.is_shutdown():
            spacenav_cmd = self.spacenav.get_joints_velocity()
            self.ecm.servo_jv(total_scale * spacenav_cmd, self.dt)

            jp_to_print = np.array2string(self.ecm.measured_jp(), precision=4, suppress_small=True)
            print(f"ECM Joints: {jp_to_print}", end="\r")
            sys.stdout.write("\033[K")
            time.sleep(self.dt)


# gui = JointGUI(
#     "ECM JOINTS",
#     4,
#     ["j0", "j1", "j2", "j3"],
#     resolution=0.00001,
#     lower_lims=[-1.0, -1.0, -1.0, -1.0],
#     upper_lims=[1.0, 1.0, 1.0, 1.0],
# )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Scale for the velocity commands. Default: 0.25",
    )
    args = parser.parse_args()

    control_interface = ControlInterface()
    control_interface.run()
