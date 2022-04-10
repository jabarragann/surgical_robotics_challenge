"""Triangulate the tail and tip of the needle
"""

import numpy as np
from numpy.linalg import inv
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCamera, AMBFStereoRig, ImageSaver, AMBFNeedle
from autonomy_utils import Frame, Logger
import rospy


class ClickyWindow:
    def __init__(self):
        self.X, self.Y = [], []
        self.count = 0
        self.close = False

    def get_pixel_values(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.count += 1
            self.X.append(x)
            self.Y.append(y)
            print(x, y)

            if self.count == 2:
                self.close = True

    def get_points_from_mouse(self, img):
        w_name = "clicky_w"
        cv2.namedWindow(w_name)
        cv2.setMouseCallback(w_name, self.get_pixel_values)
        cv2.imshow(w_name, img)

        while not self.close:
            cv2.waitKey(10)

        cv2.destroyAllWindows()

        X = np.array(self.X).reshape((-1, 1))
        Y = np.array(self.Y).reshape((-1, 1))

        return X, Y


np.set_printoptions(precision=4, suppress=True)

# If true project the ground truth to the image. Else select tip and tail with mouse
simulator_projections: bool = True

if __name__ == "__main__":

    log = Logger.Logger("autonomy_utils").log

    # Init
    rospy.init_node("image_listener")
    saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    stereo_rig = AMBFStereoRig(c)
    img_saver = ImageSaver()

    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    mtx = np.hstack((stereo_rig.camera_left.mtx, np.zeros((3, 1))))
    # Get world to right camera and projection matrix
    T_CR_W = inv(stereo_rig.get_camera_to_world_pose("right"))
    PR = mtx @ F @ T_CR_W
    # Get Left camera pose and projection matrix
    T_CL_W = inv(stereo_rig.get_camera_to_world_pose("left"))
    PL = mtx @ F @ T_CL_W

    # Get pixel locations of the tail and tip and project them to the image plane
    tip_tail_pt = needle_handle.get_tip_tail_pose()
    tip_tail_pt = needle_handle.get_current_pose() @ tip_tail_pt.T

    if simulator_projections:
        # Projection into the image plane
        pt_in_right = PR @ tip_tail_pt
        pt_in_right = pt_in_right / pt_in_right[2]
        pt_in_left = PL @ tip_tail_pt
        pt_in_left = pt_in_left / pt_in_left[2]
    else:
        # Get the points by clicking on the images
        # On left image
        img = img_saver.get_current_frame("left")
        clicky_w = ClickyWindow()
        X, Y = clicky_w.get_points_from_mouse(img)
        pt_in_left = np.ones((3, 2))
        pt_in_left[0, :] = X.squeeze()
        pt_in_left[1, :] = Y.squeeze()
        # On right image
        img = img_saver.get_current_frame("right")
        clicky_w = ClickyWindow()
        X, Y = clicky_w.get_points_from_mouse(img)
        pt_in_right = np.ones((3, 2))
        pt_in_right[0, :] = X.squeeze()
        pt_in_right[1, :] = Y.squeeze()

    # camera_handle = AMBFCamera(ambf_client=c, camera_selector="right")
    # T_CN = needle_handle.get_needle_to_camera_pose(camera_selector="right")

    log.info("input to the triangulation algorithm")
    log.info(f"point in left img\n{pt_in_left}")
    log.info(f"point in right img\n{pt_in_right}")

    # Triangulate - requires 2xn arrays, so transpose the points
    X = cv2.triangulatePoints(PL, PR, pt_in_left[:2, :], pt_in_right[:2, :])
    X /= X[3]
    # Effect of discretizing the image
    X1 = cv2.triangulatePoints(PL, PR, np.round(pt_in_left[:2, :]), np.round(pt_in_right[:2, :]))
    X1 /= X1[3]

    # Debug
    log.info(f"tip_tail_pt (True)\n{tip_tail_pt}")
    log.info(f"tip_tail_est\n{X}")
    log.info(f"error vector 0\n{X-tip_tail_pt}")
    log.info(f"error norm 0\n{np.linalg.norm((X-tip_tail_pt),axis=0)}")
    log.info(f"error vector 1\n{X1-tip_tail_pt}")
    log.info(f"error norm 1\n{np.linalg.norm((X1-tip_tail_pt),axis=0)}")

    # Display image
    img = img_saver.get_current_frame("left")
    img_pt = pt_in_left
    for i in range(img_pt.shape[1]):
        img = cv2.circle(img, (int(img_pt[0, i]), int(img_pt[1, i])), 4, (255, 0, 0), -1)

    img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    """
    It doesn't make any sense that you need the T_CW to do the triangulation. Ideally you only need the transformation between 
    the camera right and camera left.
    
    Solution: you can set the world in the left image!
    """
