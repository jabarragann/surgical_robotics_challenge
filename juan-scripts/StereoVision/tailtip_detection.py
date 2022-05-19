"""
Identify the tail-tip of the needle and the midline using the following steps:

(1) Create mask of the needle
(2) Create a bounding box 
(3) Skeletonize the needle and extract salient points in the bounding box regions.
(4) Convert the points to the full image resolution (MISSING)

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
from autonomy_utils.vision.ImageUtils import SalientPtLocator
from autonomy_utils.utils.Utils import find_correspondent_pt

import rospy
from skimage import morphology, filters
from scipy.ndimage import generic_filter

cx = AMBFCamera.cx
cy = AMBFCamera.cy

np.set_printoptions(precision=4, suppress=True)
# If true project the ground truth to the image. Else select tip and tail with mouse
simulator_projections: bool = False


def lineEnds(P):
    """find the Central pixel and one ajacent pixel is said to be a line start or line end"""
    return 255 * ((P[4] == 255) and np.sum(P) == 510)


log = Logger.Logger("autonomy_utils").log


def find_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get only the biggest contour
    max_area = 0
    max_cnt = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > max_area:
            max_cnt = c
            max_are = area

    return max_cnt
    # Draw max contour
    x, y, w, h = cv2.boundingRect(max_cnt)
    # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # start_time = time.time()
    # # ------------------------------------------------------------
    # # Initial setup
    # # ------------------------------------------------------------
    # log = Logger.Logger("autonomy_utils").log
    # rospy.init_node("image_listener")
    # c = Client("juanclient")
    # c.connect()
    # img_saver = ImageSaver()
    # needle_seg = NeedleSegmenter(ambf_client=c, log=log)

    # pt_locator = SalientPtLocator()
    # # ------------------------------------------------------------
    # # Load images and obtain mask
    # # ------------------------------------------------------------
    # left_img = img_saver.get_current_frame("left")
    # segmented_l = NeedleSegmenter.segment_needle(left_img)
    # segmented_l = needle_seg.clean_image(segmented_l, "left")
    # right_img = img_saver.get_current_frame("right")
    # segmented_r = NeedleSegmenter.segment_needle(right_img)
    # segmented_r = needle_seg.clean_image(segmented_r, "right")

    segmented_l_raw = cv2.imread("to_erase/20220113151427_l_seg.jpeg")
    max_contour = find_contours(segmented_l_raw)
    x, y, w, h = cv2.boundingRect(max_contour)
    # Expand bb
    x, y = x - 10, y - 10
    w, h = w + 10, h + 10
    segmented_l = segmented_l_raw[y : y + h, x : x + w]

    # pt_locator = SalientPtLocator()

    tic = time.perf_counter()
    data = cv2.cvtColor(segmented_l, cv2.COLOR_BGR2GRAY)
    binary = data > filters.threshold_otsu(data)
    # do the skeletonization
    skel = morphology.skeletonize(binary)
    skel = (skel * 255).astype(np.uint8)
    pt_along_axis = np.argwhere(skel > 200)
    toc = time.perf_counter()
    log.info(f"time for skeletonize {toc-tic:0.4f}")

    tic = time.perf_counter()
    # Find salient points
    tip_tail_result = generic_filter(skel, lineEnds, (3, 3))
    tip_tail_result = np.argwhere(tip_tail_result > 200)
    toc = time.perf_counter()
    log.info(f"time for finding the tip/tail {toc-tic:0.4f}")

    print(tip_tail_result)
    # pt_locator.locate_points2(segmented_l)
    # segmented_l = pt_locator.draw_solution(segmented_l)

    w_name = "final"
    skel = cv2.cvtColor(skel, cv2.COLOR_GRAY2BGR)
    for pt in pt_along_axis:
        skel[pt[0], pt[1], :] = [0, 0, 255]
        segmented_l_raw[pt[0] + y, pt[1] + x, :] = [0, 0, 255]
    for pt in tip_tail_result:
        skel[pt[0], pt[1], :] = [0, 255, 0]
        segmented_l_raw[pt[0] + y, pt[1] + x, :] = [0, 255, 0]

    segmented_l_raw = cv2.rectangle(segmented_l_raw, (x, y), (x + w, y + h), (255, 0, 255), 3)

    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    cv2.imshow(w_name, skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow(w_name, segmented_l_raw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
