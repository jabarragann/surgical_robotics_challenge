from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import rospy
import time
import numpy as np
import tf_conversions.posemath as pm

np.set_printoptions(precision=3, suppress=True, sign=" ")

rospy.init_node("dataset_collection")
c = Client("juanclient")
c.connect()

scene = Scene(c)
ambf_cam_l = Camera(c, "cameraL")
ambf_cam_r = Camera(c, "cameraR")
ambf_cam_frame = Camera(c, "CameraFrame")

counter = 0
flag = True
try:
    while flag:
        T_FC = pm.toMatrix(ambf_cam_l.get_T_c_w())
        T_WF = pm.toMatrix(ambf_cam_frame.get_T_c_w())
        print(f"pose at step {counter}")
        print(f"camera to frame\n {T_FC}")
        print(f"frame to world\n {T_WF}")
        print("\n\n\n")
        counter += 1
        time.sleep(1.5)
        if counter > 20:
            break

except Exception as e:
    print("Exit code")
    flag = False
    exit(0)
