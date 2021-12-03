"""
Least square solution to estimate the parameters of an ellipse
"""

from autonomy_utils.circle_pose_estimator import Ellipse2D
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import pickle

if __name__ == "__main__":

    width = 640
    height = 480
    X, Y = Ellipse2D.read_pts_in_file("./juan-scripts/output/sample_ellipse_01.txt", width, height)

    # Formulate and solve the least squares problem ||Ax - b ||^2
    ellipse = Ellipse2D.from_sample_points(X, Y)
    # Print the equation of the ellipse in standard form

    print("The ellipse is given by ")
    print(ellipse)

    # # Save ellipse parameters
    # # Ellipse: Ax^1 + Bxy + Cy^2 + Dx + Ey + F = 0
    # with open("./juan-scripts/output/ellipse_coefficients.txt", "w") as file:
    #     for name, param in zip(["a", "b", "c", "d", "e"], x):
    #         file.write(",".join([name, "{:0.10f}".format(param), "\n"]))
    #     file.write(",".join(["f", "{:0.10f}".format(f), "\n"]))

    # # Tests
    # x_sym = sym.Symbol("x")
    # y_sym = sym.Symbol("y")

    # ellipse = (
    #     x[0] * x_sym ** 2
    #     + x[1] * x_sym * y_sym
    #     + x[2] * y_sym ** 2
    #     + x[3] * x_sym
    #     + x[4] * y_sym
    #     + f
    # )

    # print("Sympy ellipse")
    # print(ellipse)
    # print("Evaluate expression with sympy")
    # total_error = 0
    # N = X.shape[0]
    # for i in range(N):
    #     b = [X[i, 0], Y[i, 0]]
    #     ans = ellipse.evalf(20, subs={x_sym: b[0], y_sym: b[1]})
    #     total_error += abs(ans)
    #     print(
    #         "Ellipse function evaluated at ({:0.3f},{:0.3f}) is ".format(b[0], b[1]),
    #         end="",
    #     )
    #     print(ans)

    # print("mean square error", total_error / N)
