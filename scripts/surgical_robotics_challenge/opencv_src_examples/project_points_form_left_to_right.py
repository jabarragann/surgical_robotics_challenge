"""
Script for projecting needle salient point in to the image plane

To use correctly this script make sure that:
   1) the /ambf/env/cameras/cameraL/ImageData topic is available
   2) the cameras can view the needle in the scene

Juan Antonio Barragan 
"""

from typing import Tuple
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
from surgical_robotics_challenge.simulation_manager import SimulationManager
from surgical_robotics_challenge.ecm_arm import ECM
from surgical_robotics_challenge.units_conversion import SimToSI
import cv2


np.set_printoptions(precision=4, suppress=True)


def trnorm(rot: np.ndarray):
    """Convert to proper rotation matrix
    https://petercorke.github.io/spatialmath-python/func_3d.html?highlight=trnorm#spatialmath.base.transforms3d.trnorm
    """

    unitvec = lambda x: x / np.linalg.norm(x)
    o = rot[:3, 1]
    a = rot[:3, 2]

    n = np.cross(o, a)  # N = O x A
    o = np.cross(a, n)  # (a)];
    new_rot = np.stack((unitvec(n), unitvec(o), unitvec(a)), axis=1)

    return new_rot


class ImageSub:
    def __init__(self):
        self.bridge = CvBridge()
        self.left_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )
        self.right_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraR/ImageData", Image, self.right_callback
        )

        self.left_frame = None
        self.left_ts = None

        self.right_frame = None
        self.right_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)

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


def generate_camera_intrinsics():
    # Calculate opencv camera intrinsics
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

    return intrinsic_params


def generate_needle_salient_pts():
    # needle_salient points
    theta = np.linspace(np.pi / 3, np.pi, num=8).reshape((-1, 1))

    # Scale salient points to match unit conversion in simulation manager
    radius = 0.0115 / SimToSI.linear_factor
    needle_salient = radius * np.hstack((np.cos(theta), np.sin(theta), theta * 0))

    return needle_salient


def project_to_left(simulation_manager, left_frame) -> Tuple[np.ndarray, np.ndarray]:

    # cam = simulation_manager.get_obj_handle("cameraL")
    scene = Scene(simulation_manager)  # Provides access to needle and entry/exit points
    ambf_cam_l = Camera(simulation_manager, "/ambf/env/cameras/cameraL")
    ambf_cam_frame = ECM(simulation_manager, "CameraFrame")

    T_WN = pm.toMatrix(scene.needle_measured_cp())  # Needle to world
    T_FL = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    # T_FL[:3, :3] = trnorm(T_FL[:3, :3])  # Convert to proper rotation matrix
    T_WF = pm.toMatrix(ambf_cam_frame.get_T_c_w())  # CamFrame to world

    # Get image
    img = left_frame

    # Calculate needle to left camera transformation
    T_WL = T_WF.dot(T_FL)
    T_LN = inv(T_WL).dot(T_WN)

    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    T_LN_CV2 = F.dot(T_LN)

    needle_pose_in_camera_frame = T_LN_CV2
    img = project_needle_points_to_image(needle_pose_in_camera_frame, img)

    return img, needle_pose_in_camera_frame


def project_from_left_to_right(
    simulation_manager,
    right_frame: np.ndarray,
    needle_pose_in_cam_l: np.ndarray,
):
    # Define the left camera location and orientation (AMBF convention)
    left_location = np.array([-0.02, 0.0, -0.5]) / 10
    left_look_at = np.array([0.0, 0.0, -1.0])  # Negative X-axis direction
    left_look_up = np.array([0.0, 1.0, 0.0])  # Z-axis direction

    # Define the right camera location and orientation (AMBF convention)
    right_location = np.array([0.02, 0.0, -0.5]) / 10
    right_look_at = np.array([0.0, 0.0, -1.0])  # Negative X-axis direction
    right_look_up = np.array([0.0, 1.0, 0.0])  # Z-axis direction

    # Normalize vectors
    def normalize(v):
        return v / np.linalg.norm(v)

    # Compute camera axes for left camera
    left_X = normalize(-left_look_at)
    left_Z = normalize(left_look_up)
    left_Y = normalize(np.cross(left_Z, left_X))

    # Compute camera axes for right camera
    right_X = normalize(-right_look_at)
    right_Z = normalize(right_look_up)
    right_Y = normalize(np.cross(right_Z, right_X))

    # Rotation matrices (wrt camera frame)
    R_left = np.column_stack((left_X, left_Y, left_Z))
    R_right = np.column_stack((right_X, right_Y, right_Z))

    def create_transformation_matrix(position, orientation):
        T = np.eye(4)
        T[:3, :3] = orientation  # Assuming orientation is a 3x3 rotation matrix
        # T[:3, :3] = trnorm(orientation)  # Assuming orientation is a 3x3 rotation matrix
        T[:3, 3] = position
        return T

    # left and right respect to camera frame
    T_Camera_LeftCamera = create_transformation_matrix(left_location, R_left)
    T_Camera_RightCamera = create_transformation_matrix(right_location, R_right)

    # tool pitch link to right camera
    T_opencv_ambf = np.array([[0, 0, -1, 0], [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
    T_ambf_opencv = np.linalg.inv(T_opencv_ambf)

    ####################################
    # needle pose in camera frame V1
    needle_pose_in_cam_r = (
        T_opencv_ambf
        @ np.linalg.inv(T_Camera_RightCamera)
        @ T_Camera_LeftCamera
        @ T_ambf_opencv
        @ needle_pose_in_cam_l
    )
    img = right_frame
    img = project_needle_points_to_image(needle_pose_in_cam_r, img, color=(0, 255, 0))

    ####################################
    # needle pose in camera frame V2
    scene = Scene(simulation_manager)
    ambf_cam_r = Camera(simulation_manager, "/ambf/env/cameras/cameraR")
    ambf_cam_l = Camera(simulation_manager, "/ambf/env/cameras/cameraL")
    ambf_cam_frame = ECM(simulation_manager, "CameraFrame")

    T_WN = pm.toMatrix(scene.needle_measured_cp())  # Needle to world
    T_FR = pm.toMatrix(ambf_cam_r.get_T_c_w())  # CamR to CamFrame
    # T_FR[:3, :3] = trnorm(T_FR[:3, :3])  # Convert to proper rotation matrix
    T_FL = pm.toMatrix(ambf_cam_l.get_T_c_w())  # CamL to CamFrame
    # T_FL[:3, :3] = trnorm(T_FL[:3, :3])  # Convert to proper rotation matrix
    T_WF = pm.toMatrix(ambf_cam_frame.get_T_c_w())  # CamFrame to world

    # Calculate needle to left camera transformation
    T_WR = T_WF.dot(T_FR)
    T_RN = inv(T_WR).dot(T_WN)
    # Convert AMBF camera axis to Opencv Camera axis
    F = np.array([[0, 1, 0, 0], [0, 0, -1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    T_RN_CV2 = F.dot(T_RN)

    img = project_needle_points_to_image(T_RN_CV2, img, color=(0, 255, 255))

    ##LOGGING
    print("T_FL(from simulation client): \n", T_FL)
    print("T_Camera_LeftCamera (from yaml params): \n", T_Camera_LeftCamera)

    print("T_FR: \n", T_FR)
    print("T_Camera_RightCamera: \n", T_Camera_RightCamera)
    print("conclusion there is a little rotation error when comparing the client and the yaml params")

    return img


def project_needle_points_to_image(
    needle_pose: np.ndarray, img: np.ndarray, color: Tuple[int] = (255, 0, 0)
):

    intrinsic_params = generate_camera_intrinsics()

    # Project center of the needle with OpenCv
    rvecs, _ = cv2.Rodrigues(needle_pose[:3, :3])
    tvecs = needle_pose[:3, 3]

    needle_salient = generate_needle_salient_pts()

    # Project points
    img_pt, _ = cv2.projectPoints(
        needle_salient,
        rvecs,
        tvecs,
        intrinsic_params,
        np.float32([0, 0, 0, 0, 0]),
    )

    # Display image
    for i in range(img_pt.shape[0]):
        img = cv2.circle(
            img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, color, -1
        )

    return img


def main():
    rospy.init_node("image_listener")
    simulation_manager = SimulationManager("needle_projection_ex")
    camera_subs = ImageSub()
    time.sleep(0.5)

    left = camera_subs.left_frame
    right = camera_subs.right_frame

    if left is None or right is None:
        if left is None:
            print("No left image received. Check the camera topics")
        if right is None:
            print("No right image received. Check the camera topics")
        exit(2)

    left, needle_T = project_to_left(simulation_manager, left)
    right = project_from_left_to_right(simulation_manager, right, needle_T)

    combined = np.hstack((left, right))
    window_name = "needle_projection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
