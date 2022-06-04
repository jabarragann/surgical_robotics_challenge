"""
Estimate needle pose offline script.
Offline means the ellipse coefficients are read from a txt file.
Image frame is read directly from AMBF
"""

import numpy as np
import pandas as pd
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCamera, ImageSaver, AMBFNeedle, find_closest_rotation
import rospy
from autonomy_utils.Logger import Logger

np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":
    rospy.init_node("image_listener")
    saver = ImageSaver()
    img = saver.left_frame

    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    log = Logger().log
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_selector = "left"
    camera_handle = AMBFCamera(c, camera_selector)

    # Get 3D position of the tip and tail
    needle_salient = needle_handle.get_tip_tail_pose()

    # Get needle pose wrt camera
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_coefficients("./juan-scripts/output/ellipse_coefficients_segm.txt")
    # ellipse = Ellipse2D.from_coefficients("./juan-scripts/output/ellipse_coefficients_sift.txt")
    # ellipse = Ellipse2D.from_coefficients("./juan-scripts/output/ellipse_coefficients_ideal.txt")
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
        est_normal = est_normal / np.sqrt(est_normal.dot(est_normal))
        est_x = -(tip_tail_pt[:3, 1] - circles[k].center)
        est_x = est_x / np.linalg.norm(est_x)
        est_y = np.cross(est_normal, est_x)
        est_y = est_y / np.sqrt(est_y.dot(est_y))

        # estimated pose
        pose_est = np.identity(4)
        pose_est[:3, 0] = est_x
        pose_est[:3, 1] = est_y
        pose_est[:3, 2] = est_normal
        pose_est[:3, 3] = est_center
        # re orthoganalize
        pose_est[:3, :3] = find_closest_rotation(pose_est[:3, :3])

        log.info("*" * 20)
        log.info("solution {:d}".format(k))
        log.info("*" * 20)
        log.info(f"estimated pose \n{pose_est}")
        needle_handle.pose_estimate_evaluation(pose_est, camera_selector)

    # Draw the ellipse
    # df = pd.read_csv("./juan-scripts/output/needle_segmentation_pts.txt")
    # df = pd.read_csv("./juan-scripts/output/sample_ellipse_01.txt")
    # X = df["x"].values.reshape(-1, 1)
    # Y = df["y"].values.reshape(-1, 1)

    # Debug
    log.debug(f"Ground truth TCN \n{T_CN}")
    for k in range(2):
        log.debug(f"solution {k}")
        log.debug(f"Center \n{circles[k].center}")
        log.debug(f"Normal \n{circles[k].normal}")

    for i in range(2):
        # img = np.zeros((480, 640, 3))
        img = saver.left_frame

        # Sample 3D circle
        pts = circles[i].generate_pts(40)
        # df = pd.DataFrame(pts.T, columns=["x", "y", "z"])
        # df.to_csv("./juan-scripts/output/circle{:d}.txt".format(i), index=None)

        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, camera_handle.mtx, 30, radius=3)

        # #Draw ellipse samples
        # for xp, yp in zip(X.squeeze(), Y.squeeze()):
        #     img = cv2.circle(img, (int(xp), int(yp)), radius=3, color=(0, 0, 255), thickness=-1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
