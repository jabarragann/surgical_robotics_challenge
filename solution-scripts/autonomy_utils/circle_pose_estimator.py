from __future__ import annotations
from matplotlib.pyplot import close
import numpy as np
import cv2
from numpy.linalg import norm, inv
from numpy import cos, sin, pi
from typing import List, Tuple
import pandas as pd
from skimage.measure.fit import EllipseModel


class Ellipse2D:
    def __init__(self, A, B, C, D, E, F, center=None, rx_axis=None, ry_axis=None, angle_deg=None):

        # Ellipse implicit equation: Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
        self.a_coeff = A
        self.b_coeff = B
        self.c_coeff = C
        self.d_coeff = D
        self.e_coeff = E
        self.f_coeff = F

        # Center/axis/angle representation: only if created with Opencv
        self.center = center
        self.rx_axis = rx_axis
        self.ry_axis = ry_axis
        self.angle_deg = angle_deg

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
        str_rep = "{:+10.3f}x^2 {:+10.3f}xy {:+10.3f}y^2 {:+10.3f}x {:+10.3f}y {:+10.3f} = 0".format(
            self.a_coeff,
            self.b_coeff,
            self.c_coeff,
            self.d_coeff,
            self.e_coeff,
            self.f_coeff,
        )
        return str_rep

    def plot_ellipse(self, img: np.ndarray) -> np.ndarray:
        img = cv2.ellipse(
            img,
            (int(self.center[0]), int(self.center[1])),
            (int(self.rx_axis), int(self.ry_axis)),
            self.angle_deg,
            0,
            360,
            (255, 0, 0),
            thickness=1,
        )
        return img

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
    def from_sample_points_skimage(cls: Ellipse2D, X: np.ndarray, Y: np.ndarray) -> Ellipse2D:
        xy = np.hstack([X, Y])
        ellipse = EllipseModel()
        status = ellipse.estimate(xy)
        if not status:
            raise ("Error estimating ellipse")

        x0, y0, a, b, theta = ellipse.params

        return Ellipse2D.from_principal_axis(x0, y0, a, b, theta)

    @classmethod
    def from_sample_points_cv2(cls: Ellipse2D, X: np.ndarray, Y: np.ndarray) -> Ellipse2D:
        """Estimate the ellipse coefficients from sample points int the image plane.
        use Opencv Fit ellipse method. Current implementation not working.

        Args:
            cls (Ellipse2D): [description]
            X (np.ndarray): [description]
            Y (np.ndarray): [description]

        Returns:
            Ellipse2D: [description]
        """
        contours = [np.array(np.hstack((X, Y)), dtype=np.int32)]
        center, (e_width, e_height), angle_deg = cv2.fitEllipse(contours[0])
        angle_rad = angle_deg * np.pi / 180
        rx_axis, ry_axis = (
            e_width / 2,
            e_height / 2,
        )  # rx -> axis align with x axis before rotation. The same for ry.

        a_coef, b_coef, c_coef, d_coef, e_coef, f_coef = Ellipse2D.obtain_implicit_eq(
            center, rx_axis, ry_axis, angle_rad
        )

        # Create ellipse instance
        ellipse = cls(
            a_coef,
            b_coef,
            c_coef,
            d_coef,
            e_coef,
            f_coef,
            center=list(center),
            rx_axis=rx_axis,
            ry_axis=ry_axis,
            angle_deg=angle_deg,
        )
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
    def from_principal_axis(x0: float, y0: float, a: float, b: float, theta: float) -> Ellipse2D:
        """Obtain ellipse from major/minor/principal axis representation.

        Args:
            x0 (float): center coordinate
            y0 (float): y center coordinate
            a (float): principal axis aligned to x axis (rx_axis)
            b (float): principal axis aligned to y axis (ry_axis)
            theta (float): axis rotation in rad

        Returns:
            Ellipse2D: [description]
        """

        # fmt: off
        a_coef = a ** 2 * np.sin(theta) ** 2 + b ** 2 * np.cos(theta) ** 2
        b_coef = 2 * (b ** 2 - a ** 2) * np.sin(theta) * np.cos(theta)
        c_coef = a ** 2 * np.cos(theta) ** 2 + b ** 2 * np.sin(theta) ** 2
        d_coef = -2 * a_coef * x0 - b_coef * y0
        e_coef = -b_coef * x0 - 2 * c_coef * y0
        f_coef = ( a_coef * x0 ** 2 + b_coef * x0 * y0 + c_coef * y0 ** 2 - a ** 2 * b ** 2)
        # fmt: on

        # Scale
        scale_factor = 1e6 / f_coef
        a_coef, b_coef, c_coef, d_coef, e_coef, f_coef = (
            a_coef * scale_factor,
            b_coef * scale_factor,
            c_coef * scale_factor,
            d_coef * scale_factor,
            e_coef * scale_factor,
            f_coef * scale_factor,
        )

        return Ellipse2D(a_coef, b_coef, c_coef, d_coef, e_coef, f_coef)

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
    def __init__(self, center, normal, radius):
        self.center = center
        self.radius = radius
        self.normal = normal / norm(normal)
        # self.intrinsic = intrinsic
        # Orthogonal vectors to n
        s = 0.5
        t = 0.5
        if np.isclose(self.normal[0], 0.0):
            self.a = np.array([0.5, 0.5, -(self.normal[0] * 0.5 + self.normal[1] * 0.5) / self.normal[2]])
        else:
            self.a = t * np.array([-self.normal[2] / self.normal[0], 0, 1]) + s * np.array(
                [-self.normal[1] / self.normal[0], 1, 0]
            )
        self.a /= norm(self.a)
        self.b = np.cross(self.a, self.normal)

        # a is orthogonal to n
        # l = self.normal.dot(self.a)
        assert np.isclose(abs(np.dot(self.a, self.normal)), 0.0), "a be should orthogonal to normal"
        assert np.isclose(abs(np.dot(self.b, self.normal)), 0.0), "b be should orthogonal to normal"

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

    def project_pt_to_img(self, img, intrinsic, numb_pt, radius=2):
        pts = self.generate_pts(numb_pt)
        projected = intrinsic @ pts
        projected[0, :] = projected[0, :] / projected[2, :]
        projected[1, :] = projected[1, :] / projected[2, :]
        projected[2, :] = projected[2, :] / projected[2, :]

        # img = np.zeros((480, 640, 3))
        for xp, yp in zip(projected[0, :], projected[1, :]):
            img = cv2.circle(img, (int(xp), int(yp)), radius=radius, color=(0, 255, 0), thickness=-1)

        return img

    def closest_pt_in_circ_to_pt(self, point: np.ndarray) -> np.ndarray:
        """Calculate closest point on a circle to another reference point.
        This point can be used to calculated the
        distance between the circle and the reference point.

        https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
        Args:
            point (np.ndarray): Reference point

        Returns:
            np.ndarray: 3d point on the circle.
        """

        delta = point - self.center
        dotND = np.dot(self.normal, delta)
        QmC = delta - dotND * self.normal
        lengthQmC = np.linalg.norm(QmC)
        if lengthQmC > 0:
            # crossND = np.cross(self.normal, delta)
            # radial = np.linalg.norm(crossND) - self.radius
            # sqrDistance = dotND*dotND + radial*radial
            # distance = np.sqrt(sqrDistance)
            closest_pt = self.center + self.radius * (QmC / lengthQmC)
        else:
            # Equidistant points
            # return any point in the circle
            closest_pt = self.generate_pts(1)

        return closest_pt


class CirclePoseEstimator:
    def __init__(self, ellipse: Ellipse2D, mtx: np.ndarray, focal_length: float, radius: float) -> None:
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
            Circle3D(translations[:, 0], normals[:, 0], self.radius),
            Circle3D(translations[:, 1], normals[:, 1], self.radius),
        ]


class PoseEvaluator:
    def __init__(self) -> None:
        pass
