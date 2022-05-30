from pathlib import Path
import time
import PyKDL
from autonomy_utils.Logger import Logger
from autonomy_utils.ambf_utils import AMBFNeedle, ImageSaver
from autonomy_utils.NeedlePoseEstimator import NeedlePoseEstimator
from autonomy_utils.utils.Utils import find_correspondent_pt
from autonomy_utils.DeepModels.Dice import DiceLoss
from autonomy_utils.Models.CameraModel import CameraModel

from ambf_client import Client
from surgical_robotics_challenge.ecm_arm import ECM

from geometry_msgs.msg import PoseStamped
import numpy as np
import rospy
from surgical_robotics_challenge.task_completion_report import TaskCompletionReport

log = Logger("Challenge1Solution").log


def frame_to_pose_stamped_msg(frame):
    """

    :param frame:
    :return:
    """
    msg = PoseStamped()
    msg.header.stamp = rospy.Time.now()
    msg.pose.position.x = frame.p[0]
    msg.pose.position.y = frame.p[1]
    msg.pose.position.z = frame.p[2]

    msg.pose.orientation.x = frame.M.GetQuaternion()[0]
    msg.pose.orientation.y = frame.M.GetQuaternion()[1]
    msg.pose.orientation.z = frame.M.GetQuaternion()[2]
    msg.pose.orientation.w = frame.M.GetQuaternion()[3]

    return msg


if __name__ == "__main__":

    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    camL = ECM(c, "cameraL")
    completion_report = TaskCompletionReport("JhuNeedleTeam")

    # T_e_cl --> Camera left in ecm frame
    T_e_cl = camL.get_T_c_w()

    model_path = Path("./Resources/segmentation_weights/best_model_512.pth")
    needle_pose_estimator = NeedlePoseEstimator(model_path, c)

    left_img = img_saver.get_current_frame("left")
    right_img = img_saver.get_current_frame("right")

    # T_cl_n --> Needle in camera left frame
    T_cl_n_INcv2, tip_tail_3d_est = needle_pose_estimator.estimate_pose(left_img, right_img)
    T_cl_n = CameraModel.T_ambf_cv2 @ T_cl_n_INcv2

    log.info(f"Needle pose detected \n{T_cl_n} ")
    log.info(f"Tip tail 3d position \n{tip_tail_3d_est} ")
    log.info(f"\n\n")

    vec = PyKDL.Vector(*T_cl_n[:3, 3].tolist())
    rot = PyKDL.Rotation(*T_cl_n[:3, :3].ravel().tolist())
    T_cl_n = PyKDL.Frame(rot, vec)

    # T_e_n --> Needle in ecm frame
    T_e_n = T_e_cl * T_cl_n

    # Change naming convention to be consistent with challenge evaluation script
    T_nINe_est = T_e_n
    print(f"\n{T_nINe_est}\n")

    completion_report.task_1_report(frame_to_pose_stamped_msg(T_nINe_est))

    # TODO: Transform from left camera to camera frame.
    # As a first step just do it with the ambf client to check how good is your solution with the evaluation script.
    # Then you can make this better by hard coding this values somewhere in the code.

    # # ------------------------------------------------------------
    # # Evaluate solution
    # # ------------------------------------------------------------
    # needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    # camera_selector = "left"
    # T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    # tip_tail_pt = T_CN @ needle_handle.get_tip_tail_pose().T

    # tip_tail_3d_est, tip_tail_pt = find_correspondent_pt(tip_tail_3d_est, tip_tail_pt)

    # log.info("*" * 30)
    # log.info("Triangulation evaluation")
    # log.info("*" * 30)
    # log.info(f"Error (mm) {1000*np.linalg.norm(tip_tail_3d_est-tip_tail_pt,axis=0)}")
    # log.info(f"Error (mm) {1000*np.linalg.norm(tip_tail_3d_est-tip_tail_pt,axis=0).mean():0.05f}")
    # log.info(f"\n\n")

    # log.info("*" * 30)
    # log.info("Needle pose estimate evaluation")
    # log.info("*" * 30)

    # needle_handle.pose_estimate_evaluation(pose_est, camera_selector)
