from pathlib import Path
import time
import PyKDL
from autonomy_utils.Logger import Logger
from autonomy_utils.ambf_utils import AMBFNeedle, ImageSaver
from autonomy_utils.NeedlePoseEstimator import NeedlePoseEstimator
from autonomy_utils.utils.Utils import find_correspondent_pt
from autonomy_utils.DeepModels.Dice import DiceLoss
from ambf_client import Client
from surgical_robotics_challenge.ecm_arm import ECM

import numpy as np
import rospy

log = Logger("Challenge1Solution").log

if __name__ == "__main__":

    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)

    cam_l = ECM(c, "cameraL")
    cam_r = ECM(c, "cameraR")
    ecm = ECM(c, "CameraFrame")

    print(f"ecm \n{ecm.get_T_c_w()}")
    print(f"cam_l \n{cam_l.get_T_c_w()}")
    print(f"cam_r \n{cam_r.get_T_c_w()}")

    print(f"cam_l \n{cam_l.get_T_c_w().M.GetQuaternion()}")

    # model_path = Path("./Resources/segmentation_weights/best_model_512.pth")
    # needle_pose_estimator = NeedlePoseEstimator(model_path, c)

    # left_img = img_saver.get_current_frame("left")
    # right_img = img_saver.get_current_frame("right")
