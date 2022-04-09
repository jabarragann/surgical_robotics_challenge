"""
Estimate the needle pose from a segmentation mask
"""

from tkinter import Image
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from ambf_client import Client
from autonomy_utils.ambf_utils import AMBFStereoRig, ImageSaver, AMBFCamera, AMBFNeedle, find_closest_rotation
from autonomy_utils.image_segmentation import NeedleSegmenter
from autonomy_utils.circle_pose_estimator import Ellipse2D, CirclePoseEstimator
from autonomy_utils import Logger
import rospy
import spatialmath as sm
import matplotlib.pyplot as plt

cx = AMBFCamera.cx
cy = AMBFCamera.cy


def get_points_from_mask(img, segmenter: NeedleSegmenter):

    mask = segmenter.segment_needle(img)
    clean_mask_rgb = segmenter.clean_image(mask, "left", ambf_client=c, log=log)
    clean_mask = cv2.cvtColor(clean_mask_rgb, cv2.COLOR_BGR2GRAY)

    ########################################
    ## Obtain needle bounding rect
    contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    # x, y, w, h = rect
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    ########################################
    ## Get pixels corresponding to the needle
    needle_pts = []
    for i in range(rect[1], rect[1] + rect[3]):
        for j in range(rect[0], rect[0] + rect[2]):
            if clean_mask[i, j] == 255:
                needle_pts.append([j, i])  # Save x,y positions. Origin in the left top corner
                img[i, j] = [0, 255, 0]
    needle_pts = np.array(needle_pts)
    X = needle_pts[:, 0].reshape((-1, 1))
    Y = needle_pts[:, 1].reshape((-1, 1))

    return X, Y


class ClickyWindow:
    def __init__(self):
        self.X, self.Y = [], []

    def get_pixel_values(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.X.append(x)
            self.Y.append(y)
            print(x, y)

    def get_points_from_mouse(self, img):
        w_name = "clicky_w"
        cv2.namedWindow(w_name)
        cv2.setMouseCallback(w_name, self.get_pixel_values)
        cv2.imshow(w_name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        X = np.array(self.X).reshape((-1, 1))
        Y = np.array(self.Y).reshape((-1, 1))

        return X, Y


if __name__ == "__main__":

    ########################################
    ## Environment initialization

    rospy.init_node("main_node")
    c = Client("juanclient")
    log = Logger.Logger().log
    c.connect()
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_selector = "left"
    camera_handle = AMBFCamera(c, camera_selector)
    stereo_rig_handle = AMBFStereoRig(ambf_client=c)
    segmenter_handler = NeedleSegmenter.from_handler(needle_handle, stereo_rig_handle)
    ########################################
    ## Read and create mask
    img_saver = ImageSaver()
    img = img_saver.get_current_frame("left")
    # X, Y = get_points_from_mask(img)

    clicky_w = ClickyWindow()
    X, Y = clicky_w.get_points_from_mouse(img)

    ########################################
    ## Estimate ellipse
    ellipse = Ellipse2D.from_sample_points_skimage(X - cx, Y - cy)
    log.info(f"The ellipse is given by {ellipse}")

    ########################################
    ## Estimate pose
    estimator = CirclePoseEstimator(ellipse, camera_handle.mtx, camera_handle.focal_length, needle_handle.radius)
    circles = estimator.estimate_pose()

    ########################################
    ## Evaluate
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    needle_salient = needle_handle.get_tip_tail_pose()
    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    ## Groudn truth transformation matrx
    needle_center = T_CN[:3, 3]
    needle_x_axis = T_CN[:3, 0]
    needle_y_axis = T_CN[:3, 1]
    needle_normal = T_CN[:3, 2]

    estimated_solutions = []
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
        # re orthogonalize
        pose_est[:3, :3] = find_closest_rotation(pose_est[:3, :3])
        estimated_solutions.append(pose_est)
        log.info("*" * 20)
        log.info("solution {:d}".format(k))
        log.info("*" * 20)
        log.info(f"estimated pose \n{pose_est}")
        needle_handle.pose_estimate_evaluation(pose_est, camera_selector)

    ########################################
    ## Draw estimated circles
    for i in range(2):
        img = img_saver.left_frame
        # Sample 3D circle
        pts = circles[i].generate_pts(40)
        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, camera_handle.mtx, 30, radius=3)
        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows

    ########################################
    ## Draw pose solutions
    # T0 = sm.SE3(T_CN)
    # T1 = sm.SE3(estimated_solutions[0])
    # T2 = sm.SE3(estimated_solutions[1])

    # T0.plot(frame="0", color="black")
    # T1.plot(frame="1", color="green")
    # T1.plot(frame="1", color="red")
    # plt.show()

    ## Show selected points
    # w_name = "mask_w"
    # cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(w_name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
