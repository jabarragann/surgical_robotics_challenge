from __future__ import annotations
import numpy as np
import cv2
from numpy.linalg import norm, inv
from numpy import cos, sin, pi
from typing import List, Tuple
import pandas as pd
from sympy.logic.boolalg import anf_coeffs


class Ellipse2D:
    def __init__(self, A, B, C, D, E, F):

        # Ellipse implicit equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        self.a_coeff = A
        self.b_coeff = B
        self.c_coeff = C
        self.d_coeff = D
        self.e_coeff = E
        self.f_coeff = F

        self.parameter_vector = [
            self.a_coeff,
            self.b_coeff,
            self.c_coeff,
            self.d_coeff,
            self.e_coeff,
            self.f_coeff,
        ]

    def get_c_matrix(self):
        c_mat = np.ones((3, 3))
        c_mat[0, 0] = self.a_coeff
        c_mat[1, 1] = self.c_coeff
        c_mat[2, 2] = self.f_coeff
        c_mat[0, 1] = c_mat[1, 0] = self.b_coeff / 2
        c_mat[0, 2] = c_mat[2, 0] = self.d_coeff / 2
        c_mat[1, 2] = c_mat[2, 1] = self.e_coeff / 2
        return c_mat

    def parameters_to_txt(self: Ellipse2D, file: str) -> None:
        # Ellipse implicit equation: Ax^1 + Bxy + Cy^2 + Dx + Ey + F = 0
        x = [self.a_coeff, self.b_coeff, self.c_coeff, self.d_coeff, self.e_coeff, self.f_coeff]
        with open(file, "w") as file_handle:
            for name, param in zip(["a", "b", "c", "d", "e", "f"], x):
                file_handle.write(",".join([name, "{:0.10f}".format(param), "\n"]))

    def set_coefficients(self, A, B, C, D, E, F) -> None:
        self.a_coeff = A
        self.b_coeff = B
        self.c_coeff = C
        self.d_coeff = D
        self.e_coeff = E
        self.f_coeff = F

    def __str__(self):
        str_rep = "{0:.6}x^2 +{1:.6}xy + {2:.6}y^2 + {3:.6}x + {4:.6}y + {5:.6} = 0".format(
            self.a_coeff,
            self.b_coeff,
            self.c_coeff,
            self.d_coeff,
            self.e_coeff,
            self.f_coeff,
        )
        return str_rep

    @classmethod
    def from_sample_points(cls: Ellipse2D, X: np.ndarray, Y: np.ndarray) -> Ellipse2D:
        """Estimate the ellipse coefficients from sample points int the image plane.
        The estimation of the coefficients is done using the least squares solution for Ax = b

        [X^2 XY Y^2 X Y] @ [A|  = F
                            B|
                            C|
                            D|

        Args:
            cls (Ellipse2D): [description]
            X (np.ndarray): x coordiante of the sample points. Shape (N,1)
            Y (np.ndarray): y coordinate of the sample points. Shape (N,1)

        Returns:
            Ellipse2D: estimated ellipse
        """
        # Formulate and solve the least squares problem ||Ax - b ||^2

        F = -1e6  # Ellipse constant
        A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
        b = np.ones_like(X) * F
        x = np.linalg.lstsq(A, b, rcond=None)[0].squeeze()

        ellipse = cls(x[0], x[1], x[2], x[3], x[4], -F)
        return ellipse

    @classmethod
    def from_coefficients(cls: Ellipse2D, coefficients_file: str) -> Ellipse2D:
        d = {
            "a": None,
            "b": None,
            "c": None,
            "d": None,
            "e": None,
            "f": None,
        }
        with open(coefficients_file, "r") as f1:
            for line in f1.readlines():
                vals = line.split(sep=",")
                d[vals[0]] = float(vals[1])

        return Ellipse2D(d["a"], d["b"], d["c"], d["d"], d["e"], d["f"])

    @staticmethod
    def read_pts_in_file(file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Read sample points from a txt file and center them using the width and height of the image.


        Args:
            file (str): Path of the file

        Returns:
            tuple[np.ndarray, np.ndarray]: two arrays containing the X,Y coordinates of the sample points. Each array
            has a shape (N,1) where the `N` is the number of sample points
        """
        df = pd.read_csv(file)

        X = df["x"].values.reshape(-1, 1)
        Y = df["y"].values.reshape(-1, 1)

        return X, Y


class Circle3D:
    def __init__(self, center, normal, radius, intrinsic):
        self.center = center
        self.radius = radius
        self.normal = normal / norm(normal)
        self.intrinsic = intrinsic
        # Orthogonal vectors to n
        s = 0.5
        t = 0.5
        self.a = t * np.array([-self.normal[2] / self.normal[0], 0, 1]) + s * np.array(
            [-self.normal[1] / self.normal[0], 1, 0]
        )
        self.a /= norm(self.a)
        self.b = np.cross(self.a, self.normal)

        # a is orthogonal to n
        # l = self.normal.dot(self.a)

    def generate_pts(self, N):
        """Generate `N` sample point from the parametric representation of the 3D circle

        Args:
            numb_pt ([type]): [description]

        Returns:
            [type]: [description]
        """
        pts = np.zeros((3, N))
        theta = np.linspace(0, 2 * pi, N).reshape(-1, 1)
        pts = self.center + self.radius * cos(theta) * self.a + self.radius * sin(theta) * self.b
        pts = pts.T
        return pts

    def project_pt_to_img(self, img, intrinsic, numb_pt):
        pts = self.generate_pts(numb_pt)
        projected = intrinsic @ pts
        projected[0, :] = projected[0, :] / projected[2, :]
        projected[1, :] = projected[1, :] / projected[2, :]
        projected[2, :] = projected[2, :] / projected[2, :]

        # img = np.zeros((480, 640, 3))
        for xp, yp in zip(projected[0, :], projected[1, :]):
            img = cv2.circle(img, (int(xp), int(yp)), radius=1, color=(0, 255, 0), thickness=-1)

        return img


class CirclePoseEstimator:
    def __init__(
        self, ellipse: Ellipse2D, mtx: np.ndarray, focal_length: float, radius: float
    ) -> None:
        """[summary]

        Args:
            ellipse (Ellipse2D): [description]
            mtx (np.ndarray): [description]
            focal_length (float): [description]
            radius (float): [description]
        """

        self.ellipse = ellipse
        self.mtx = mtx
        self.focal_length = focal_length
        self.radius = radius

        self.c_mat = ellipse.get_c_matrix()
        # Normalize ellipse matrix with focal length
        self.c_mat[0, 0] = self.c_mat[0, 0]
        self.c_mat[0, 1] = self.c_mat[0, 1]
        self.c_mat[0, 2] = self.c_mat[0, 2] / (focal_length)
        self.c_mat[1, 0] = self.c_mat[1, 0]
        self.c_mat[1, 1] = self.c_mat[1, 1]
        self.c_mat[1, 2] = self.c_mat[1, 2] / (focal_length)
        self.c_mat[2, 0] = self.c_mat[2, 0] / (focal_length)
        self.c_mat[2, 1] = self.c_mat[2, 1] / (focal_length)
        self.c_mat[2, 2] = self.c_mat[2, 2] / (focal_length * focal_length)

    def estimate_pose(self) -> List[Circle3D]:
        """[summary]

        Returns:
            List[Circle3D]: [description]
        """

        # Calculate Eigen vectors
        ret, W, V = cv2.eigen(self.c_mat)
        V = V.transpose()

        e1 = W[0, 0]
        e2 = W[1, 0]
        e3 = W[2, 0]

        S1 = [+1, +1, +1, +1, -1, -1, -1, -1]
        S2 = [+1, +1, -1, -1, +1, +1, -1, -1]
        S3 = [+1, -1, +1, -1, +1, -1, +1, -1]

        g = np.sqrt((e2 - e3) / (e1 - e3))
        h = np.sqrt((e1 - e2) / (e1 - e3))

        # Calculate two possible solutions
        translations = np.zeros((3, 2))
        normals = np.zeros((3, 2))
        projections = np.zeros((2, 2))

        k = 0
        for i in range(8):
            z0 = S3[i] * (e2 * self.radius) / np.sqrt(-e1 * e3)
            Tx = S2[i] * e3 / e2 * h
            Ty = 0.0
            Tz = -S1[i] * e1 / e2 * g
            Nx = S2[i] * h
            Ny = 0.0
            Nz = -S1[i] * g

            t = z0 * V @ np.array([Tx, Ty, Tz]).reshape((3, 1))
            n = V @ np.array([Nx, Ny, Nz]).reshape((3, 1))

            if t[2] > 0 and n[2] < 0:
                translations[:, k] = t.reshape(-1)
                normals[:, k] = n.reshape(-1)
                pc = self.mtx @ t
                projections[:, k] = [pc[0] / pc[2], pc[1] / pc[2]]
                k += 1

        return [
            Circle3D(translations[:, 0], normals[:, 0], self.radius, self.mtx),
            Circle3D(translations[:, 1], normals[:, 1], self.radius, self.mtx),
        ]


class PoseEvaluator:
    def __init__(self) -> None:
        pass
