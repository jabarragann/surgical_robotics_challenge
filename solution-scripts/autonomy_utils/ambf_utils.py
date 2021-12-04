from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image
import time
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
import cv2
import tf_conversions.posemath as pm
import numpy as np
from numpy.linalg import inv, norm
import pandas as pd


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


class AMBFNeedle:
    def __init__(self, ambf_client) -> None:

        self.c = ambf_client
        self.scene = Scene(self.c)
        self.ambf_cam_l = Camera(self.c, "cameraL")
        self.ambf_cam_frame = Camera(self.c, "CameraFrame")

        self.radius = 0.1018

    def get_tip_tail_pose(self) -> np.ndarray:
        """Get 3D positions of the tip and the tail w.r.t the needle local frame.
        The needle is parametrized by a circle of radius `radius`. The tip of the needle is
        located at the angle pi/3 and the tail at  pi.

        Returns:
            np.ndarray: [description]
        """
        # Get 3D position of the tip and tail
        theta = np.array([np.pi / 3, np.pi]).reshape((2, 1))
        radius = 0.1018
        needle_salient = radius * np.hstack(
            (np.cos(theta), np.sin(theta), theta * 0, np.ones((2, 1)) / radius)
        )
        return needle_salient

    def get_current_pose(self) -> np.ndarray:
        """Generates the needle current pose with respect to the left camera frame. The resulting matrix
            uses the opencv convention instead of the AMBF convention.

        Returns:
            np.ndarray: 4x4 transformation matrix representing the needle pose with respect to the camera

        """

        T_WN = pm.toMatrix(self.scene.needle_measured_cp())  # Needle to world
        T_FC = pm.toMatrix(self.ambf_cam_l.get_T_c_w())  # CamL to CamFrame
        T_WF = pm.toMatrix(self.ambf_cam_frame.get_T_c_w())  # CamFrame to world

        T_WC = T_WF @ T_FC
        T_CN = inv(T_WC) @ T_WN

        # Convert AMBF camera axis to Opencv Camera axis
        F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
        T_CN = F @ T_CN

        return T_CN

    def sample_3d_pts(self, N: int) -> np.ndarray:
        """Sample `N` 3D points of the needle on the needle local coordinate frame

        Args:
            N (int): [description]

        Returns:
            np.ndarray: Needle 3D points
        """

        theta = np.linspace(np.pi / 3, np.pi, num=N).reshape((-1, 1))
        needle_salient = self.radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))
        return needle_salient


class AMBFCameras:
    def __init__(self) -> None:
        # compute camera intrinsics
        self.fvg = 1.2
        self.width = 640
        self.height = 480
        self.f = self.height / (2 * np.tan(self.fvg / 2))

        intrinsic_params = np.zeros((3, 3))
        intrinsic_params[0, 0] = self.f
        intrinsic_params[1, 1] = self.f
        intrinsic_params[0, 2] = self.width / 2
        intrinsic_params[1, 2] = self.height / 2
        intrinsic_params[2, 2] = 1.0
        self.mtx = intrinsic_params

        self.focal_length = (self.mtx[0, 0] + self.mtx[1, 1]) / 2

    def project_points(self, T_CW: np.ndarray, obj_3d_pt: np.ndarray) -> np.ndarray:

        rvecs, _ = cv2.Rodrigues(T_CW[:3, :3])
        tvecs = T_CW[:3, 3]

        img_pt, _ = cv2.projectPoints(
            obj_3d_pt,
            rvecs,
            tvecs,
            self.mtx,
            np.float32([0, 0, 0, 0, 0]),
        )
        return img_pt

    @staticmethod
    def save_projected_points(file_path: str, img_pt):
        results_df = pd.DataFrame(columns=["id", "x", "y"])
        for i in range(img_pt.shape[0]):
            # Save pts
            results_df = results_df.append(
                pd.DataFrame(
                    [[i, int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])]],
                    columns=["id", "x", "y"],
                )
            )
        results_df.to_csv(file_path, index=None)
