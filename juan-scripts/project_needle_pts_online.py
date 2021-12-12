"""
Project needle points into the image and save those points for ellipse parameter estimation

"""
import cv2
import numpy as np
from ambf_client import Client
import time
import ambf_client
import rospy
import pandas as pd
from autonomy_utils.ambf_utils import ImageSaver, AMBFCamera, AMBFNeedle

np.set_printoptions(precision=3)

if __name__ == "__main__":
    camera_selector = "left"
    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    img = img_saver.get_current_frame(camera_selector)

    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)
    needle_handle = AMBFNeedle(ambf_client=c)
    camera_handle = AMBFCamera(ambf_client=c, camera_selector=camera_selector)

    # Get 3D position of the tip and tail
    needle_salient = needle_handle.get_tip_tail_pose()

    # Get needle pose wrt camera
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    # Sample point on the needle
    needle_pts = needle_handle.sample_3d_pts(8)

    # project points
    img_pt = camera_handle.project_points(T_CN, needle_pts)

    # Equivalent to cv2.projectPoints
    intrinsic_params_2 = np.hstack((camera_handle.mtx, np.zeros((3, 1))))
    img_pt_2 = intrinsic_params_2 @ T_CN @ np.array([0, 0, 0, 1]).reshape((4, 1))
    img_pt_2 = img_pt_2 / img_pt_2[2]

    # Print information
    print("intrinsic")
    print(camera_handle.mtx)
    print("T_CN. Transform from the needle to cam")
    print(T_CN)
    # print("T_WN. Transform from needle to world")
    # print(T_WN)
    # print("T_WC. Transform from camera to world")
    # print(T_WC)
    print("Projected center")
    print(img_pt[0, 0])
    print(img_pt_2.reshape(-1))

    AMBFCamera.save_projected_points("./juan-scripts/output/sample_ellipse_01.txt", img_pt)

    # Display image
    for i in range(img_pt.shape[0]):
        img = cv2.circle(img, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 3, (255, 0, 0), -1)

    window_n = "output_window"
    cv2.namedWindow(window_n, cv2.WINDOW_NORMAL)
    cv2.imshow(window_n, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
