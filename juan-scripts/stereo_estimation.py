"""Triangulate the tail and tip of the needle
"""

import numpy as np
from numpy.linalg import inv
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCamera, AMBFStereoRig, ImageSaver, AMBFNeedle
import rospy

np.set_printoptions(precision=6)


if __name__ == "__main__":
    # Init
    rospy.init_node("image_listener")
    saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c)
    stereo_rig = AMBFStereoRig(c)

    mtx = np.hstack((stereo_rig.camera_left.mtx, np.zeros((3, 1))))
    # Get world to right camera and projection matrix
    T_CR_W = inv(stereo_rig.get_camera_to_world_pose("right"))
    PR = mtx @ T_CR_W
    # Get Left camera pose and projection matrix
    T_CL_W = inv(stereo_rig.get_camera_to_world_pose("left"))
    # Get pixel locations of the tail and tip and project them to the image plane
    tip_tail_pt = needle_handle.get_tip_tail_pose()
    tip_tail_pt = needle_handle.get_current_pose() @ tip_tail_pt.T
    pt_in_right = PR @ tip_tail_pt
    pt_in_right = pt_in_right / pt_in_right[2]

    print(tip_tail_pt)
    print(pt_in_right)
    # pt_in_right = stereo_rig.camera_right.project_points(T_CR_W, tip_tail_pt)
    # pt_in_left = stereo_rig.camera_left.project_points(T_CL_W, tip_tail_pt)
    # Triangulate

    # Display image
    img_saver = ImageSaver()
    img = img_saver.get_current_frame("right")
    img_pt = pt_in_right
    for i in range(img_pt.shape[1]):
        img = cv2.circle(img, (int(img_pt[0, i]), int(img_pt[1, i])), 3, (255, 0, 0), -1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
