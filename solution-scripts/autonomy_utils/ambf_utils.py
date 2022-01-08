from re import I
from cv_bridge import CvBridge, CvBridgeError
import rospy
from rospy import client
from sensor_msgs.msg import Image
import time
from surgical_robotics_challenge.scene import Scene
from surgical_robotics_challenge.camera import Camera
import cv2
import tf_conversions.posemath as pm
import numpy as np
from numpy.linalg import inv, norm
import pandas as pd
from pathlib import Path

def find_closest_rotation(matrix:np.ndarray) -> np.ndarray:
    """ Find closest rotation to the input matrix
    Algorithm from https://stackoverflow.com/questions/23080791/eigen-re-orthogonalization-of-rotation-matrix/23083722

    Args:
        matrix (np.ndarray): (3x3) rotation matrix 

    Returns:
        np.ndarray: [description]
    """

    #Method 1
    # def normalize(x):
    #     return x/np.sqrt(x.dot(x))

    # x = matrix[:3,0]
    # y = matrix[:3,1]
    # z = matrix[:3,2]
    
    # error = x.dot(z)
    # new_x = x-(error/2)*z
    # new_z = z-(error/2)*x
    # new_y = np.cross(new_z,new_x) 

    # new_matrix = np.zeros_like(matrix)
    # new_matrix[:3,0] = normalize(new_x)
    # new_matrix[:3,1] = normalize(new_y)
    # new_matrix[:3,2] = normalize(new_z)

    #Method 2
    u, s, vh = np.linalg.svd(matrix, full_matrices=True)
    new_matrix = u @ vh

    assert np.isclose(np.linalg.det(new_matrix),1.0), "Normalization procedure failed"

    return new_matrix 

class ImageSaver:
    def __init__(self):
        """This class will not work unless you initialize a ROS topic

        rospy.init_node("image_listener")
        """
        self.bridge = CvBridge()
        self.left_cam_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )
        self.right_cam_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraR/ImageData", Image, self.right_callback
        )
        self.left_frame = np.zeros((640, 480, 3))
        self.left_ts = None
        self.right_frame = np.zeros((640, 480, 3))
        self.right_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.6)

    def get_current_frame(self, camera_selector: str) -> np.ndarray:
        if camera_selector == "left":
            return self.left_frame
        elif camera_selector == "right":
            return self.right_frame
        else:
            raise ValueError("camera selector should be either 'left' or 'right'")

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def right_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.right_frame = cv2_img
            self.right_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)

    def save_frame(self, camera_selector: str, path: Path):
        if camera_selector not in ["left", "right"]:
            ValueError("camera selector error")

        img = self.left_frame if camera_selector == "left" else self.right_frame
        ts = self.left_ts if camera_selector == "left" else self.right_ts
        name = camera_selector + "_frame" + ".jpeg"
        # Save frame
        cv2.imwrite(str(path / name), img)  ## Opencv does not work with pathlib


class AMBFNeedle:
    def __init__(self, ambf_client, logger) -> None:

        self.c = ambf_client
        self.logger = logger
        self.scene = Scene(self.c)
        self.ambf_cam_l = Camera(self.c, "cameraL")
        self.ambf_cam_r = Camera(self.c, "cameraR")
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
        """Get the needle current position with respect to its parent in the scene graph (this is the world.)

        Returns:
            np.ndarray: [description]
        """
        return pm.toMatrix(self.scene.needle_measured_cp())  # Needle to world

    def get_needle_to_camera_pose(self, camera_selector: str) -> np.ndarray:
        """Generates the needle current pose with respect to the selected camera coordinate frame. The resulting matrix
            uses the opencv convention instead of the Opencv convention.

        Args:
            camera_selector (str): either "left" or "right"

        Returns:
            np.ndarray: 4x4 transformation matrix representing the needle pose with respect to the camera
        """
        """

        Returns:
            np.ndarray: 

        """
        if camera_selector == "left":
            T_FC = pm.toMatrix(self.ambf_cam_l.get_T_c_w())  # CamL to CamFrame
        elif camera_selector == "right":
            T_FC = pm.toMatrix(self.ambf_cam_r.get_T_c_w())  # CamR to CamFrame
        else:
            raise ValueError("camera selector should be either 'left' or 'right'")

        T_WN = pm.toMatrix(self.scene.needle_measured_cp())  # Needle to world
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

    def pose_estimate_evaluation(self, pose_est: np.ndarray, camera_selector: str) -> None:
        T_CN = self.get_needle_to_camera_pose(camera_selector)

        #Get salient_points
        tip_tail_pt =  self.get_tip_tail_pose()

        # Ground truth
        needle_center = T_CN[:3, 3]
        needle_x_axis = T_CN[:3, 0]
        needle_y_axis = T_CN[:3, 1]
        needle_normal = T_CN[:3, 2]

        # Estimated pose - this is not a rigid transformation!!
        est_x = pose_est[:3, 0]
        est_y = pose_est[:3, 1]
        est_normal = pose_est[:3, 2]
        est_center = pose_est[:3, 3]

        # Check that rotation matrix has determinant of one.
        if not np.isclose(np.linalg.det(pose_est[:3,:3]),1.0):
            self.logger.warning("determinant of the rotation matrix is not 1.")


        #Metrics
        x_ang_diff = np.arccos(needle_x_axis.dot(est_x)) * 180 / np.pi
        y_ang_diff = np.arccos(needle_y_axis.dot(est_y))* 180 / np.pi
        normal_ang_diff = np.arccos(needle_normal.dot(est_normal))* 180 / np.pi

        tip_tail_true = T_CN @ tip_tail_pt.T
        tip_tail_est = pose_est @ tip_tail_pt.T

        # fmt: on
        self.logger.info("x-axis angle error (deg):     {:10.2f}".format(x_ang_diff))
        self.logger.info("y-axis angle error (deg):     {:10.2f}".format(y_ang_diff))
        self.logger.info("Normal angle error (deg):     {:10.2f}".format(normal_ang_diff))
        self.logger.info("Center MSE error (cm):        {:10.2f}".format(100*np.linalg.norm(needle_center - est_center)))
        self.logger.info("tip    MSE error (cm):        {:10.2f}".format(100*np.linalg.norm(tip_tail_true[:,0]-tip_tail_est[:,0])))
        self.logger.info("tail   MSE error (cm):        {:10.2f}".format(100*np.linalg.norm(tip_tail_true[:,1]-tip_tail_est[:,1])))
        self.logger.info("plane vect dot normal:        {:10.2f}".format(est_normal.dot(needle_x_axis)))
        # fmt: off


class AMBFCamera:
    fvg = 1.2
    width = 1920
    height = 1080
    cx = width / 2
    cy = height / 2

    def __init__(self, ambf_client, camera_selector: str) -> None:
        """AMBF camera handler

        Args:
            ambf_client ([type]): [description]
            camera_name (str): either 'left' or 'right'
        """

        if camera_selector not in ["left", "right"]:
            raise ValueError("camera selector must be either 'left' or 'right'")

        # AMBF handlers
        self.c = ambf_client
        self.scene = Scene(self.c)
        camera_name = "cameraL" if camera_selector == "left" else "cameraR"
        self.ambf_cam = Camera(self.c, camera_name)

        # Initialize extrinsic
        self.T_W_C = self.ambf_cam.get_T_c_w()

        # Calculate intrinsic

        self.f = self.height / (2 * np.tan(self.fvg / 2))

        intrinsic_params = np.zeros((3, 3))
        intrinsic_params[0, 0] = self.f
        intrinsic_params[1, 1] = self.f
        intrinsic_params[0, 2] = self.width / 2
        intrinsic_params[1, 2] = self.height / 2
        intrinsic_params[2, 2] = 1.0
        self.mtx = intrinsic_params

        self.focal_length = (self.mtx[0, 0] + self.mtx[1, 1]) / 2

    def get_current_pose(self) -> np.ndarray:
        """Get the camera current position with respect to its parent in the scene graph (this is the stereo rig frame.)

        Returns:
            np.ndarray: [description]
        """
        return pm.toMatrix(self.ambf_cam.get_T_c_w())

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
                    [[i, img_pt[i, 0, 0], img_pt[i, 0, 1]]],
                    columns=["id", "x", "y"],
                )
            )
        path = Path(file_path)
        if not path.parent.exists():
            path.parent.mkdir()

        results_df.to_csv(file_path, index=None)


class AMBFStereoRig:
    def __init__(self, ambf_client) -> None:
        """Camera model for AMBF cameras. Since both cameras in the stereo rig have the same in intrinsic parameters,
        a single instance of this class works for both the right and left camera of the stereo rig.

        Intrinsic parameters will be the same but extrinsic parameters will be different
        """
        self.camera_left = AMBFCamera(ambf_client, "left")
        self.camera_right = AMBFCamera(ambf_client, "right")
        self.ambf_cam_frame = Camera(ambf_client, "CameraFrame")

    def get_intrinsics(self, camera_selector: str) -> np.ndarray:
        if camera_selector == "left":
            return self.camera_left.mtx
        elif camera_selector == "right":
            return self.camera_right.mtx
        else:
            raise ValueError("camera selector should be either 'left' or 'right'")

    def get_camera_to_world_pose(self, camera_selector: str):
        """Get camera to world pose.

            T_WC = T_WF @ T_FC

        Args:
            camera_selector (str): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        if camera_selector == "left":
            return self.get_current_pose() @ self.camera_left.get_current_pose()
        elif camera_selector == "right":
            return self.get_current_pose() @ self.camera_right.get_current_pose()

        else:
            raise ValueError("camera selector should be either 'left' or 'right'")

    def get_current_pose(self) -> np.ndarray:
        """Get the stereo rig current position with respect to its parent in the scene graph (this is the world frame.)

        Returns:
            np.ndarray: [description]
        """
        return pm.toMatrix(self.ambf_cam_frame.get_T_c_w())
