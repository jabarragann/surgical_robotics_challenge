import json
import cv2
import numpy as np
from numpy.linalg import inv
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
from ambf_client import Client
import time
import tf_conversions.posemath as pm
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospy
import pandas as pd

np.set_printoptions(precision=3)


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
    # Connect to AMBF and setup image suscriber
    rospy.init_node("image_listener")
    saver = ImageSaver()

    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)

    scene = Scene(c)
    ambf_cam_l = Camera(c, "cameraL")
    ambf_cam_frame = Camera(c, "CameraFrame")

    # Calculate intrinsics
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

    # Get pose for the needle and the camera
    T_WN = pm.toMatrix(scene.needle_measured_cp())  # Needle to world
    T_FC = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    T_WF = pm.toMatrix(ambf_cam_frame.get_T_c_w())  # CamFrame to world

    # Get image
    img = saver.left_frame

    # Calculate Needle to Camera  transformation
    # T_WC = T_WF @(T_FC)
    # T_CN = inv(T_WC) @ (T_WN)

    T_WC = T_WF.dot(T_FC)
    T_CN = inv(T_WC).dot(T_WN)

    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    T_CN = F @ T_CN

    # Project center of the needle with OpenCv
    rvecs, _ = cv2.Rodrigues(T_CN[:3, :3])
    tvecs = T_CN[:3, 3]

    # needle_salient = np.float32([[0, 0, 0], [-0.1018, 0, 0], [0, 0.1018, 0]])
    theta = np.linspace(np.pi / 3, np.pi, num=8).reshape((-1, 1))
    radius = 0.1018
    needle_salient = radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))

    img_pt, _ = cv2.projectPoints(
        needle_salient,
        rvecs,
        tvecs,
        intrinsic_params,
        np.float32([0, 0, 0, 0, 0]),
    )

    # Equivalent to cv2.projectPoints
    intrinsic_params_2 = np.hstack((intrinsic_params, np.zeros((3, 1))))
    img_pt_2 = intrinsic_params_2 @ T_CN @ np.array([0, 0, 0, 1]).reshape((4, 1))
    img_pt_2 = img_pt_2 / img_pt_2[2]

    # Print information
    print("intrinsic")
    print(intrinsic_params)
    print("T_WN. Transform from needle to world")
    print(T_WN)
    print("T_WC. Transform from camera to world")
    print(T_WC)
    print("T_CN. Transform from the needle to cam")
    print(T_CN)
    print("Projected center")
    print(img_pt[0, 0])
    print(img_pt_2.reshape(-1))

    # Display image
    results_df = pd.DataFrame(columns=["id", "x", "y"])
    for i in range(img_pt.shape[0]):
        img = cv2.circle(
            img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, (255, 0, 0), -1
        )

        # Save pts
        results_df = results_df.append(
            pd.DataFrame(
                [[i, int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])]],
                columns=["id", "x", "y"],
            )
        )
    results_df.to_csv("./sample_ellipse_01.txt", index=None)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
