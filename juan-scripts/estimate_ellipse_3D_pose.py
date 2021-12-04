from os import W_OK
import pickle
import numpy as np
import pandas as pd
import math as m
import cv2
from numpy.linalg import norm
from numpy import cos, sin, pi
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
from numpy.linalg import inv
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import time
import tf_conversions.posemath as pm
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator

np.set_printoptions(precision=6)


class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )

        self.left_frame = None
        self.left_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)


if __name__ == "__main__":
    rospy.init_node("image_listener")
    saver = ImageSaver()
    img = saver.left_frame

    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)

    scene = Scene(c)
    ambf_cam_l = Camera(c, "cameraL")
    ambf_cam_frame = Camera(c, "CameraFrame")
    # Ground truth
    radius = 0.1018

    # Read camera parameters
    fvg = 1.2
    width = 640
    height = 480
    f = height / (2 * np.tan(fvg / 2))

    intrinsic_params = np.zeros((3, 3))
    intrinsic_params[0, 0] = f
    intrinsic_params[1, 1] = f
    intrinsic_params[0, 2] = width / 2
    intrinsic_params[1, 2] = height / 2
    intrinsic_params[2, 2] = 1.0
    mtx = intrinsic_params

    focal_length = (mtx[0, 0] + mtx[1, 1]) / 2

    # Get 3D position of the tip and tail
    theta = np.array([np.pi / 3, np.pi]).reshape((2, 1))
    radius = 0.1018
    needle_salient = radius * np.hstack(
        (np.cos(theta), np.sin(theta), theta * 0, np.ones((2, 1)) / radius)
    )

    T_WN = pm.toMatrix(scene.needle_measured_cp())  # Needle to world
    T_FC = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    T_WF = pm.toMatrix(ambf_cam_frame.get_T_c_w())  # CamFrame to world

    T_WC = T_WF.dot(T_FC)
    T_CN = inv(T_WC).dot(T_WN)

    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    T_CN = F @ T_CN

    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_coefficients("./juan-scripts/output/ellipse_coefficients.txt")
    estimator = CirclePoseEstimator(ellipse, mtx, focal_length, radius)
    circles = estimator.estimate_pose()

    print("algorithm summary")
    print("camera_matrix")
    print(mtx)
    print("focal length {:0.4f}".format(focal_length))
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
    df = pd.read_csv("./juan-scripts/output/sample_ellipse_01.txt")
    X = df["x"].values.reshape(-1, 1)
    Y = df["y"].values.reshape(-1, 1)

    for i in range(2):
        # img = np.zeros((480, 640, 3))
        img = saver.left_frame

        # Sample 3D circle
        pts = circles[i].generate_parametric(30)
        df = pd.DataFrame(pts.T, columns=["x", "y", "z"])
        df.to_csv("circle{:d}.txt".format(i), index=None)

        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, mtx, 30)

        # #Draw ellipse samples
        for xp, yp in zip(X.squeeze(), Y.squeeze()):
            img = cv2.circle(img, (int(xp), int(yp)), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
