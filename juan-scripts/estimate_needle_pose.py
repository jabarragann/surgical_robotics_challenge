import numpy as np
import pandas as pd
import cv2
from ambf_client import Client
import time
from autonomy_utils.circle_pose_estimator import Circle3D, Ellipse2D, CirclePoseEstimator
from autonomy_utils.ambf_utils import AMBFCameras, ImageSaver, AMBFNeedle
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
    camera_handle = AMBFCameras()

    # Get 3D position of the tip and tail
    needle_salient = needle_handle.get_tip_tail_pose()

    # Get needle pose wrt camera
    T_CN = needle_handle.get_current_pose()

    tip_tail_pt = T_CN @ needle_salient.T
    plane_vect = tip_tail_pt[:3, 0] - tip_tail_pt[:3, 1]

    # Obtain points to estimate ellipse
    needle_pts = needle_handle.sample_3d_pts(8)
    projected_needle_pts = camera_handle.project_points(T_CN, needle_pts).squeeze()
    X, Y = projected_needle_pts[:, 0].reshape(-1, 1), projected_needle_pts[:, 1].reshape(-1, 1)
    X -= camera_handle.cx
    Y -= camera_handle.cy

    # Normalize ellipse coefficients
    ellipse = Ellipse2D.from_sample_points(X, Y)
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

    for i in range(2):
        # img = np.zeros((480, 640, 3))
        img = saver.left_frame

        # Sample 3D circle
        pts = circles[i].generate_parametric(30)
        df = pd.DataFrame(pts.T, columns=["x", "y", "z"])
        df.to_csv("./juan-scripts/output/circle{:d}.txt".format(i), index=None)

        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, camera_handle.mtx, 30)

        # #Draw ellipse samples
        for xp, yp in zip(X.squeeze(), Y.squeeze()):
            img = cv2.circle(img, (int(xp), int(yp)), radius=2, color=(0, 0, 255), thickness=-1)

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
