"""
Estimate the needle pose from a segmentation mask
"""

import cv2
from pathlib import Path
import numpy as np
import pandas as pd
from ambf_client import Client
from autonomy_utils.ambf_utils import ImageSaver, AMBFCamera, AMBFNeedle
from autonomy_utils.image_segmentation import segment_needle, clean_image
from autonomy_utils.circle_pose_estimator import Ellipse2D, CirclePoseEstimator
from autonomy_utils import Logger
import rospy


cx = AMBFCamera.cx
cy = AMBFCamera.cy

if __name__ == "__main__":
    # Environment initialization
    rospy.init_node("main_node")
    c = Client("juanclient")
    log = Logger.Logger().log
    c.connect()
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_selector = "left"
    camera_handle = AMBFCamera(c, camera_selector)

    # Read and create mask
    img_saver = ImageSaver()
    img = img_saver.get_current_frame("left")

    mask = segment_needle(img)
    clean_mask_rgb = clean_image(mask, "left", ambf_client=c, log=log)
    clean_mask = cv2.cvtColor(clean_mask_rgb, cv2.COLOR_BGR2GRAY)

    # Obtain needle bounding rect
    contours, hierarchy = cv2.findContours(clean_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.boundingRect(contours[0])
    # x, y, w, h = rect
    # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    # cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    # Get pixels corresponding to the needle
    needle_pts = []
    for i in range(rect[1], rect[1] + rect[3]):
        for j in range(rect[0], rect[0] + rect[2]):
            if clean_mask[i, j] == 255:
                needle_pts.append([j, i])  # Save x,y positions. Origin in the left top corner
                img[i, j] = [0, 255, 0]
    needle_pts = np.array(needle_pts)
    X = needle_pts[:, 0].reshape((-1, 1))
    Y = needle_pts[:, 1].reshape((-1, 1))

    ## Estimate ellipse
    ellipse = Ellipse2D.from_sample_points_skimage(X - cx, Y - cy)
    log.info(f"The ellipse is given by {ellipse}")

    ## Estimate pose
    estimator = CirclePoseEstimator(
        ellipse, camera_handle.mtx, camera_handle.focal_length, needle_handle.radius
    )
    circles = estimator.estimate_pose()

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

    # # Show selected points
    # w_name = "mask_w"
    # cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    # cv2.imshow(w_name, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
