"""
Estimate needle pose offline script.
Online means the ellipse coefficients are calculated from the AMBF image.

"""
import numpy as np
import pandas as pd
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCamera, ImageSaver, AMBFNeedle
import rospy
from autonomy_utils.Logger import Logger

np.set_printoptions(precision=6)

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

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_sample_points(X - camera_handle.cx, Y - camera_handle.cy)
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
        est_center = circles[k].center

        est_normal = circles[k].normal
        est_normal = est_normal/np.sqrt(est_normal.dot(est_normal))
        est_x = -(tip_tail_pt[:3, 1] - circles[k].center)
        est_x = est_x/np.linalg.norm(est_x)
        est_y = np.cross(est_normal,est_x) 
        est_y = est_y/np.sqrt(est_y.dot(est_y))

        #estimated pose
        pose_est = np.ones((4,4))
        pose_est[:3,0] = est_x
        pose_est[:3,1] = est_y
        pose_est[:3,2] = est_normal
        pose_est[:3,3] = est_center
        
        # fmt: on
        # print("solution {:d}".format(k))
        # print("x-axis MSE error:      {:6.4f}".format(np.linalg.norm(needle_x_axis- est_x)))
        # print("y-axis MSE error:      {:6.4f}".format(np.linalg.norm(needle_y_axis- est_y)))
        # print("Normal MSE error:      {:6.4f}".format(np.linalg.norm(needle_normal- est_normal)))
        # print("Center MSE error:      {:6.4f}".format(np.linalg.norm(needle_center - est_center)))
        # print("plane vect dot normal: {:6.4f}".format(circles[k].normal.dot(plane_vect)))
        # fmt: off
        log.info("solution {:d}".format(k))
        needle_handle.pose_estimate_evaluation(pose_est,camera_selector)
    
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
