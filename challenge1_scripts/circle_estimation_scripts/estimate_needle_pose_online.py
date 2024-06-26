"""
Estimate needle pose offline script.
Online means the ellipse coefficients are calculated from the AMBF image.

"""
import numpy as np
import pandas as pd
import cv2
from ambf_client import Client
import time
from autonomy_utils.ambf_utils import AMBFCamera, ImageSaver, AMBFNeedle
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
import rospy
from autonomy_utils.Logger import Logger

np.set_printoptions(precision=3)

""" Estimate needle pose and calculate MSE error
"""

if __name__ == "__main__":
    # Init
    camera_selector = "left"
    rospy.init_node("image_listener")
    saver = ImageSaver()
    c = Client("juanclient")
    log = Logger().log 
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_handle = AMBFCamera(c,camera_selector)

    # Get 3D position of the tip and tail
    needle_salient = needle_handle.get_tip_tail_pose()

    # Get needle pose wrt camera - ground truth
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    # Obtain points to estimate ellipse
    needle_pts = needle_handle.sample_3d_pts(8)
    projected_needle_pts = camera_handle.project_points(T_CN, needle_pts).squeeze()
    X, Y = projected_needle_pts[:, 0].reshape(-1, 1), projected_needle_pts[:, 1].reshape(-1, 1)
    X, Y = X.astype(np.int32),Y.astype(np.int32) 

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_sample_points_skimage(X - camera_handle.cx, Y - camera_handle.cy)
    log.info(f"Ellipse parameters: {str(ellipse)}")
    estimator = CirclePoseEstimator(
        ellipse, camera_handle.mtx, camera_handle.focal_length, needle_handle.radius
    )
    circles = estimator.estimate_pose()

    # Ground truth
    needle_center = T_CN[:3, 3]
    needle_x_axis = T_CN[:3, 0]
    needle_y_axis = T_CN[:3, 1]
    needle_normal = T_CN[:3, 2]

    for k in range(2):
        pose_est = AMBFNeedle.circle2needlepose(circles[k], tip_tail_pt[:3, 1])
        
        # fmt: on
        # print("solution {:d}".format(k))
        # print("x-axis MSE error:      {:6.4f}".format(np.linalg.norm(needle_x_axis- est_x)))
        # print("y-axis MSE error:      {:6.4f}".format(np.linalg.norm(needle_y_axis- est_y)))
        # print("Normal MSE error:      {:6.4f}".format(np.linalg.norm(needle_normal- est_normal)))
        # print("Center MSE error:      {:6.4f}".format(np.linalg.norm(needle_center - est_center)))
        # print("plane vect dot normal: {:6.4f}".format(circles[k].normal.dot(plane_vect)))
        # fmt: off
        log.info("*"*20)
        log.info("solution {:d}".format(k))
        log.info("*"*20)
        log.info(f"estimated pose \n{pose_est}")
        needle_handle.pose_estimate_evaluation(pose_est,camera_selector)
    
    #Debug - statements
    log.debug(f"Ground truth TCN \n{T_CN}")
    for k in range(2):
        log.debug(f"solution {k}")
        log.debug(f"Center \n{circles[k].center}")
        log.debug(f"Normal \n{circles[k].normal}")

    #Show projected ellipses
    for i in range(2):
        img = saver.get_current_frame(camera_selector)
        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, camera_handle.mtx, 30)
        # Draw ellipse samples
        for xp, yp in zip(X.squeeze(), Y.squeeze()):
            img = cv2.circle(img, (int(xp), int(yp)), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
