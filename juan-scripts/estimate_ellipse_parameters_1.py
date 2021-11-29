"""
Least square solution to estimate the parameters of an ellipse
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sympy as sym
import pickle

if __name__ == "__main__":

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

    mtx = intrinsic_params
    cx, cy = mtx[0, 2], mtx[1, 2]

    print("camera matrix")
    print(mtx)

    df = pd.read_csv("./juan-scripts/output/sample_ellipse_01.txt")

    X = df["x"].values.reshape(-1, 1) - cx
    Y = df["y"].values.reshape(-1, 1) - cy

    # Formulate and solve the least squares problem ||Ax - b ||^2
    f = -1e6  # Ellipse constant
    A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    b = np.ones_like(X) * (-f)
    x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

    # Print the equation of the ellipse in standard form
    print(x * 1e6)

    print("The ellipse is given by ")
    print(
        "{0:.8}x^2 +{1:.8}xy + {2:.8}y^2 + {3:.8}x + {4:.8}y + {5:.8} = 0".format(
            x[0], x[1], x[2], x[3], x[4], f
        )
    )

    # Save ellipse parameters
    # Ellipse: ax^2 + by^2 +cxy +dx + ey + f = 0
    with open("./juan-scripts/output/ellipse_coefficients.txt", "w") as file:
        for name, param in zip(["a", "b", "c", "d", "e"], x):
            file.write(",".join([name, "{:0.10f}".format(param), "\n"]))
        file.write(",".join(["f", "{:0.10f}".format(f), "\n"]))

    # Tests
    x_sym = sym.Symbol("x")
    y_sym = sym.Symbol("y")

    ellipse = (
        x[0] * x_sym ** 2
        + x[1] * x_sym * y_sym
        + x[2] * y_sym ** 2
        + x[3] * x_sym
        + x[4] * y_sym
        + f
    )

    print("Sympy ellipse")
    print(ellipse)
    print("Evaluate expression with sympy")
    total_error = 0
    for i in range(df.shape[0]):
        b = [X[i, 0], Y[i, 0]]
        ans = ellipse.evalf(20, subs={x_sym: b[0], y_sym: b[1]})
        total_error += abs(ans)
        print(
            "Ellipse function evaluated at ({:0.3f},{:0.3f}) is ".format(b[0], b[1]),
            end="",
        )
        print(ans)

    print("mean square error", total_error / df.shape[0])
