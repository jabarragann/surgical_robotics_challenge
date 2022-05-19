"""Triangulate the tail and tip of the needle from segmentation mask
"""

from autonomy_utils.vision.ImageSegmentator import NeedleSegmenter
import numpy as np
from numpy.linalg import inv
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCamera, AMBFStereoRig, ImageSaver, AMBFNeedle
from autonomy_utils import Frame, Logger
from autonomy_utils.vision import ImageUtils
from autonomy_utils.utils.Utils import find_correspondent_pt

import rospy


np.set_printoptions(precision=4, suppress=True)
# If true project the ground truth to the image. Else select tip and tail with mouse
simulator_projections: bool = False

if __name__ == "__main__":

    # ------------------------------------------------------------
    # Initial setup
    # ------------------------------------------------------------
    log = Logger.Logger("autonomy_utils").log
    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    stereo_rig = AMBFStereoRig(c)
    img_saver = ImageSaver()
    needle_seg = NeedleSegmenter(ambf_client=c, log=log)

    # ------------------------------------------------------------
    # Load images and obtain mask
    # ------------------------------------------------------------
    left_img = img_saver.get_current_frame("left")
    segmented_l = NeedleSegmenter.segment_needle(left_img)
    segmented_l = needle_seg.clean_image(segmented_l, "left")
    right_img = img_saver.get_current_frame("right")
    segmented_r = NeedleSegmenter.segment_needle(right_img)
    segmented_r = needle_seg.clean_image(segmented_r, "right")

    # ------------------------------------------------------------
    # Calculate tip/tail from images
    # ------------------------------------------------------------
    ## TODO: Optimize tip/tail detection algorithm too slow!
    start = time.time()
    # Left
    # tip_tail_pix_l = [(1408, 681), (1204, 816)]
    # img, tip_tail_pix_l, points_along_needle = ImageUtils.locate_points2(segmented_l)
    points_along_needle_l, tip_tail_pix_l, cnt, bb = ImageUtils.ImageProcessing.calculate_needle_salient_points(
        segmented_l
    )
    log.info(f"tip/tail in left {tip_tail_pix_l}\n")
    # Right
    # tip_tail_pix_r = [(1435, 686), (1209, 824)]
    # img, tip_tail_pix_r, points_along_needle = ImageUtils.locate_points2(segmented_r)
    points_along_needle_r, tip_tail_pix_r, cnt, bb = ImageUtils.ImageProcessing.calculate_needle_salient_points(
        segmented_r
    )
    log.info(f"tip/tail in right {tip_tail_pix_r}\n")

    log.info(f"total time {time.time()-start}")
    # TODO: create a matching algorithms. Match the first point in left img
    # TODO: with the closest point in right img.

    # ------------------------------------------------------------
    # Calculate projection matrices for both cameras
    # ------------------------------------------------------------
    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    mtx = np.hstack((stereo_rig.camera_left.mtx, np.zeros((3, 1))))
    # Get world to right camera and projection matrix
    T_CR_W = inv(stereo_rig.get_camera_to_world_pose("right"))
    PR = mtx @ F @ T_CR_W
    # Get Left camera pose and projection matrix
    T_CL_W = inv(stereo_rig.get_camera_to_world_pose("left"))
    PL = mtx @ F @ T_CL_W

    # ------------------------------------------------------------
    # Triangulate points
    # ------------------------------------------------------------
    tip_tail_pt = needle_handle.get_tip_tail_pose()  # In needle frame
    tip_tail_pt = needle_handle.get_current_pose() @ tip_tail_pt.T  # In world frame

    to_arr = lambda x: np.array(x).transpose().astype(np.float64)
    pt_in_l = to_arr(tip_tail_pix_l)
    pt_in_r = to_arr(tip_tail_pix_r)

    X1 = cv2.triangulatePoints(PL, PR, pt_in_l, pt_in_r)
    X1 /= X1[3]

    # ------------------------------------------------------------
    # Compare against ground truth
    # ------------------------------------------------------------
    # Get ground truth locations of the tail and tip and project them to the image plane
    tip_tail_pt = needle_handle.get_tip_tail_pose()  # In needle frame
    tip_tail_pt = needle_handle.get_current_pose() @ tip_tail_pt.T  # In world frame

    X1, tip_tail_pt = find_correspondent_pt(X1, tip_tail_pt)
    log.info("estimation of the tip and tail")
    log.info(X1)
    log.info("ground truth")
    log.info(tip_tail_pt)
    log.info(f"Error (m) {np.linalg.norm(X1-tip_tail_pt,axis=0)}")
    log.info(f"Error (m) {np.linalg.norm(X1-tip_tail_pt,axis=0).mean():0.05f}")

    # ------------------------------------------------------------
    # Show results
    # ------------------------------------------------------------
    # Draw tip and tail
    segmented_l = cv2.circle(segmented_l, tuple(tip_tail_pix_l[0]), 10, (255, 0, 0), -1)
    segmented_l = cv2.circle(segmented_l, tuple(tip_tail_pix_l[1]), 10, (255, 0, 0), -1)
    segmented_r = cv2.circle(segmented_r, tuple(tip_tail_pix_r[0]), 10, (255, 0, 0), -1)
    segmented_r = cv2.circle(segmented_r, tuple(tip_tail_pix_r[1]), 10, (255, 0, 0), -1)
    # Combine left and right into a single frame to display
    segmented_l = cv2.resize(segmented_l, (640, 480), interpolation=cv2.INTER_AREA)
    segmented_r = cv2.resize(segmented_r, (640, 480), interpolation=cv2.INTER_AREA)
    final = np.hstack((segmented_l, segmented_r))
    cv2.imshow("final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
