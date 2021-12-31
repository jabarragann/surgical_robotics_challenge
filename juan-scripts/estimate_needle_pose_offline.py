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
from autonomy_utils.ambf_utils import AMBFCamera, ImageSaver, AMBFNeedle
import rospy

np.set_printoptions(precision=6)


if __name__ == "__main__":
    rospy.init_node("image_listener")
    saver = ImageSaver()
    img = saver.left_frame

    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c)
    camera_handle = AMBFCamera(c, "left")

    # Get 3D position of the tip and tail
    needle_salient = needle_handle.get_tip_tail_pose()

    # Get needle pose wrt camera
    T_CN = needle_handle.get_current_pose()

    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_coefficients("./juan-scripts/output/ellipse_coefficients_segm.txt")
    estimator = CirclePoseEstimator(
        ellipse, camera_handle.mtx, camera_handle.focal_length, needle_handle.radius
    )
    circles = estimator.estimate_pose()

    print("algorithm summary")
    print("camera_matrix")
    print(camera_handle.mtx)
    print("focal length {:0.4f}".format(camera_handle.focal_length))
    print("ellipse c matrix")
    print(estimator.c_mat)
    # print("eigen values")
    # print(W)
    # print("eigen vectors")
    # print(V)

    for k in range(2):
        print("solution {:d}".format(k))
        print("pose")
        print(circles[k].center)
        print("normal")
        print(circles[k].normal)
        print("plane vect dot normal")
        print(circles[k].normal.dot(plane_vect))

    # Draw the ellipse
    # df = pd.read_csv("./juan-scripts/output/needle_segmentation_pts.txt")
    df = pd.read_csv("./juan-scripts/output/sample_ellipse_01.txt")

    X = df["x"].values.reshape(-1, 1)
    Y = df["y"].values.reshape(-1, 1)

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
