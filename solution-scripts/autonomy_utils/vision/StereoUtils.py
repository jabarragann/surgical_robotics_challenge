import time
from autonomy_utils.vision.ImageSegmenter import NeedleSegmenter
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


class StereoLocator:
    def __init__(self, client) -> None:
        """Stereo estimator for tip and tail.

        Parameters
        ----------
        client : _type_
            Ambf client
        """
        # Class containing the camera's pose information.
        self.stereo_rig = AMBFStereoRig(client)
        # Camera intrinsic parameters. Parameters are assumed the same for left and right cameras
        self.mtx = np.hstack((self.stereo_rig.camera_left.mtx, np.zeros((3, 1))))
        # Transformation to Opencv convention
        self.F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])

    def locate_tip_tail_3d(self, tip_tail_left, tip_tail_right) -> np.ndarray:
        """Give the pixel locations of tha tail and tail.

        Parameters
        ----------
        tip_tail_left : _type_
            _description_
        tip_tail_right : _type_
            _description_
        mtx : _type_
            _description_

        Returns
        -------
        tip_tail_3d: np.ndarray
            Return a 2x4 array with the 3d location of the tip and tail of the needle. Each point is represented
            with homogenous coordinates.
        """

        # ------------------------------------------------------------
        # Calculate projection matrices for both cameras - Set left camera as the origin
        # ------------------------------------------------------------
        # Get Left camera pose and projection matrix
        T_CL_CL = np.identity(4)
        PL = self.mtx @ self.F @ T_CL_CL
        # Get left cam to right camera transform and and then the projection matrix
        T_F_CL = self.stereo_rig.camera_left.get_current_pose()  # Left cam to frame
        T_F_CR = self.stereo_rig.camera_right.get_current_pose()  # Right cam to frame
        T_CR_CL = np.linalg.inv(T_F_CR) @ T_F_CL
        PR = self.mtx @ self.F @ T_CR_CL

        # ------------------------------------------------------------
        # Triangulate points
        # ------------------------------------------------------------
        to_arr = lambda x: np.array(x).transpose().astype(np.float64)
        pt_in_l = to_arr(tip_tail_left)
        pt_in_r = to_arr(tip_tail_right)

        X1 = cv2.triangulatePoints(PL, PR, pt_in_l, pt_in_r)
        X1 /= X1[3]
        X1 = self.F @ X1  # Transform triangulated points to opencv convention

        return X1


if __name__ == "__main__":
    start_time = time.time()
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
    stereo_locator = StereoLocator(c)

    cx = AMBFCamera.cx
    cy = AMBFCamera.cy

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
    start = time.time()
    # Left
    points_along_needle_l, tip_tail_pix_l, cnt, bb = ImageUtils.ImageProcessing.calculate_needle_salient_points(
        segmented_l
    )
    # Right
    points_along_needle_r, tip_tail_pix_r, cnt, bb = ImageUtils.ImageProcessing.calculate_needle_salient_points(
        segmented_r
    )
    log.info(f"total time {time.time()-start}")

    ########################################
    ## Estimate pose
    X = np.array(points_along_needle_l)[:, 0].reshape(-1, 1)
    Y = np.array(points_along_needle_l)[:, 1].reshape(-1, 1)
    ellipse = Ellipse2D.from_sample_points_skimage(X - cx, Y - cy)
    log.info(f"The ellipse is given by {ellipse}")

    camera_selector = "left"
    camera_handle = AMBFCamera(c, camera_selector)
    estimator = CirclePoseEstimator(ellipse, camera_handle.mtx, camera_handle.focal_length, needle_handle.radius)
    circles = estimator.estimate_pose()

    # ------------------------------------------------------------
    # Stereo calculation
    # ------------------------------------------------------------
    X1 = stereo_locator.locate_tip_tail_3d(tip_tail_pix_l, tip_tail_pix_r)

    # Select circle that minimizes distance to triangulated points.
    dist_list = []
    for k in range(2):
        t_list = []
        for i in range(2):
            closest1 = circles[k].closest_pt_in_circ_to_pt(X1[:3, i])
            dist1 = np.linalg.norm(X1[:3, i] - closest1)
            t_list.append(dist1)
        # log.info(t_list)
        dist_list.append((t_list[0] + t_list[1]) / 2)

    dist_list = np.array(dist_list)
    selected_circle = np.argmin(dist_list)
    log.info(f"distance to estimated circles {dist_list}")

    # ------------------------------------------------------------
    # Compare against ground truth
    # ------------------------------------------------------------
    # Get ground truth locations of the tail and tip and project them to the image plane
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    tip_tail_pt = T_CN @ needle_handle.get_tip_tail_pose().T

    X1, tip_tail_pt = find_correspondent_pt(X1, tip_tail_pt)
    log.info("*" * 30)
    log.info("Triangulation evaluation")
    log.info("*" * 30)
    log.info("estimation of the tip and tail")
    log.info(X1)
    log.info("ground truth")
    log.info(tip_tail_pt)
    log.info(f"Error (m) {np.linalg.norm(X1-tip_tail_pt,axis=0)}")
    log.info(f"Error (m) {np.linalg.norm(X1-tip_tail_pt,axis=0).mean():0.05f}")

    log.info("*" * 30)
    log.info("Needle pose estimate evaluation")
    log.info("*" * 30)

    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    tip_tail_pt = T_CN @ needle_handle.get_tip_tail_pose().T
    for k in range(2):

        pose_est = AMBFNeedle.circle2needlepose(circles[k], tip_tail_pt[:3, 1])

        log.info("*" * 20)
        s = "(Best)" if k == selected_circle else ""
        log.info(f"solution {k:d} {s}")
        log.info("*" * 20)
        log.info(f"estimated pose \n{pose_est}")
        needle_handle.pose_estimate_evaluation(pose_est, camera_selector)

    log.info(f"Time to estimate pose {time.time() - start_time}")

    # ------------------------------------------------------------
    # Show results
    # ------------------------------------------------------------
    # Draw tip and tail
    segmented_l = cv2.circle(segmented_l, tuple(tip_tail_pix_l[0]), 10, (255, 0, 0), -1)
    segmented_l = cv2.circle(segmented_l, tuple(tip_tail_pix_l[1]), 10, (255, 0, 0), -1)
    segmented_r = cv2.circle(segmented_r, tuple(tip_tail_pix_r[0]), 10, (255, 0, 0), -1)
    segmented_r = cv2.circle(segmented_r, tuple(tip_tail_pix_r[1]), 10, (255, 0, 0), -1)
    points_along_needle_l = np.array(points_along_needle_l)
    for i in range(points_along_needle_l.shape[0]):
        segmented_l = cv2.circle(segmented_l, tuple(points_along_needle_l[i, :]), 3, (0, 255, 0), -1)

    # Combine left and right into a single frame to display
    segmented_l = cv2.resize(segmented_l, (640, 480), interpolation=cv2.INTER_AREA)
    segmented_r = cv2.resize(segmented_r, (640, 480), interpolation=cv2.INTER_AREA)
    final = np.hstack((segmented_l, segmented_r))
    cv2.imshow("final", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
