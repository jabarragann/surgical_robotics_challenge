import json
import cv2
import numpy as np
from numpy.linalg import inv
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import time
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.units_conversion import SimToSI


np.set_printoptions(precision=3, suppress=True)


def main():
    rospy.init_node("image_listener")
    simulation_manager = SimulationManager("needle_projection_ex")
    time.sleep(0.5)

    scene = Scene(simulation_manager)  # Provides access to needle and entry/exit points
    ambf_cam_l = Camera(simulation_manager, "/ambf/env/cameras/cameraL")
    ambf_cam_r = Camera(simulation_manager, "/ambf/env/cameras/cameraR")
    ambf_cam_frame = ECM(simulation_manager, "CameraFrame")


    T_FL = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    T_FR = pm.toMatrix(ambf_cam_r.get_T_c_w())  # CamL to CamFrame

    print("T_FL: \n", T_FL)
    print("T_FR: \n", T_FR)


if __name__ == "__main__":
    main()