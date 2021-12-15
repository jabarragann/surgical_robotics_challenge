"""
Compare ellipse fitting methods
"""

from autonomy_utils.circle_pose_estimator import Ellipse2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import pickle
from autonomy_utils.ambf_utils import AMBFCamera
import cv2

if __name__ == "__main__":

    cx = AMBFCamera.cx
    cy = AMBFCamera.cy
    width = AMBFCamera.width
    height = AMBFCamera.height

    # fmt: off
    X, Y = Ellipse2D.read_pts_in_file("./juan-scripts/output/sample_ellipse_01.txt")
    # X, Y = Ellipse2D.read_pts_in_file( "./juan-scripts/output/needle_segmentation_pts.txt")
    # fmt: on

    # Build ellipse
    ellipse_1 = Ellipse2D.from_sample_points(X - cx, Y - cy)
    ellipse_2 = Ellipse2D.from_sample_points_cv2(X - cx, Y - cy)
    # Print the equation of the ellipse
    print(f"The ellipse1 is given by {ellipse_1}")
    print(f"The ellipse2 is given by {ellipse_2}")

    img = np.zeros((height, width, 3))
    ellipse_2.plot_ellipse(img)
    for i in range(X.shape[0]):
        img = cv2.circle(img, (X[i, 0], Y[i, 0]), radius=2, color=(0, 0, 255), thickness=-1)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
