"""
Least square solution to estimate the parameters of an ellipse
"""

from autonomy_utils.circle_pose_estimator import Ellipse2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import pickle
from autonomy_utils.ambf_utils import AMBFCamera

if __name__ == "__main__":

    cx = AMBFCamera.cx
    cy = AMBFCamera.cy
    # fmt: off
    # X, Y = Ellipse2D.read_pts_in_file("./juan-scripts/output/sample_ellipse_01.txt")
    X, Y = Ellipse2D.read_pts_in_file( "./juan-scripts/output/needle_segmentation_pts.txt")
    # fmt: on

    # Build ellipse
    ellipse = Ellipse2D.from_sample_points(X - cx, Y - cy)
    # Print the equation of the ellipse
    print(f"The ellipse is given by {ellipse}")

    # # Save ellipse parameters
    parameters_file = "./juan-scripts/output/ellipse_coefficients.txt"
    ellipse.parameters_to_txt(parameters_file)

    x = ellipse.parameter_vector

    # Tests
    x_sym = sym.Symbol("x")
    y_sym = sym.Symbol("y")

    ellipse = (
        x[0] * x_sym ** 2
        + x[1] * x_sym * y_sym
        + x[2] * y_sym ** 2
        + x[3] * x_sym
        + x[4] * y_sym
        + x[5]
    )

    print("Sympy ellipse")
    print(ellipse)
    print("Evaluate expression with sympy")
    total_error = 0
    N = X.shape[0]
    for i in range(N):
        b = [X[i, 0], Y[i, 0]]
        ans = ellipse.evalf(20, subs={x_sym: b[0], y_sym: b[1]})
        total_error += abs(ans)
        print("Ellipse function evaluated at ({:0.3f},{:0.3f}) is {:0.3f}".format(b[0], b[1], ans))

    print("mean square error", total_error / N)
