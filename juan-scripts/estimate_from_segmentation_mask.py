"""
Estimate the needle pose from a segmentation mask
"""

import time
from tkinter import Image
import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from ambf_client import Client
from autonomy_utils.ambf_utils import AMBFStereoRig, ImageSaver, AMBFCamera, AMBFNeedle
from autonomy_utils.vision.ImageSegmenter import NeedleSegmenter
from autonomy_utils.circle_pose_estimator import Ellipse2D, CirclePoseEstimator
from autonomy_utils import Logger
from autonomy_utils.vision import ImageUtils

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
    img_saver = ImageSaver()
    c.connect()
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_selector = "left"
    camera_handle = AMBFCamera(c, camera_selector)
    stereo_rig_handle = AMBFStereoRig(ambf_client=c)
    # segmenter_handler = NeedleSegmenter.from_handler(needle_handle, stereo_rig_handle)
    needle_seg = NeedleSegmenter(ambf_client=c, log=log)

    time.sleep(0.3)

    img = img_saver.get_current_frame(camera_selector)
    log.info(img.dtype)
    log.info(img.shape)
    segmented_l = NeedleSegmenter.segment_needle(img)
    segmented_l = needle_seg.clean_image(segmented_l, camera_selector=camera_selector)

    ########################################
    ## Get needle salient point to estimate pose
    ########################################
    # Method 1: get points from mask
    # X, Y = get_points_from_mask(img)
    # Method 2: Use a clicky window
    clicky_w = ClickyWindow()
    X, Y = clicky_w.get_points_from_mouse(img)
    # Method 3: automatic detection
    # img, tip_tail_pix_l, points_along_needle = ImageUtils.locate_points(segmented_l, pt_along_needle=35)
    # log.info(f"tip/tail in left {tip_tail_pix_l}\n")
    # X = np.array(points_along_needle)[:, 0].reshape(-1, 1)
    # Y = np.array(points_along_needle)[:, 1].reshape(-1, 1)

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
    tip_tail_pt = T_CN @ needle_handle.get_tip_tail_pose().T

    estimated_solutions = []
    for k in range(2):
        pose_est = AMBFNeedle.circle2needlepose(circles[k], tip_tail_pt[:3, 1])

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
