from os import W_OK
import pickle
import numpy as np
import pandas as pd
import math as m
import cv2
from numpy.linalg import norm
from numpy import cos, sin, pi
from cv_bridge import CvBridge, CvBridgeError
import rospy
from sensor_msgs.msg import Image

np.set_printoptions(precision=6)


class ImageSaver:
    def __init__(self):
        self.bridge = CvBridge()
        self.img_subs = rospy.Subscriber(
            "/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback
        )

        self.left_frame = None
        self.left_ts = None

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError as e:
            print(e)


class ellipse_2d:
    def __init__(self):
        # Ellipse: ax^2 + by^2 +cxy +dx + ey + f = 0
        self.a_coeff = 0.1
        self.b_coeff = 0.0
        self.c_coeff = 0.0
        self.d_coeff = 0.0
        self.e_coeff = 0.0
        self.f_coeff = 0.0

    def get_c_matrix(self):
        c_mat = np.ones((3, 3))
        c_mat[0, 0] = self.a_coeff
        c_mat[1, 1] = self.c_coeff
        c_mat[2, 2] = self.f_coeff
        c_mat[0, 1] = c_mat[1, 0] = self.b_coeff / 2
        c_mat[0, 2] = c_mat[2, 0] = self.d_coeff / 2
        c_mat[1, 2] = c_mat[2, 1] = self.e_coeff / 2
        return c_mat

    def load_coeff(self, file):
        d = {
            "a": self.a_coeff,
            "b": self.b_coeff,
            "c": self.c_coeff,
            "d": self.d_coeff,
            "e": self.e_coeff,
            "f": self.f_coeff,
        }
        with open(file, "r") as f1:
            for line in f1.readlines():
                vals = line.split(sep=",")
                d[vals[0]] = float(vals[1])
        self.a_coeff = d["a"]
        self.b_coeff = d["b"]
        self.c_coeff = d["c"]
        self.d_coeff = d["d"]
        self.e_coeff = d["e"]
        self.f_coeff = d["f"]


class Circle_3d:
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

    def generate_parametric(self, numb_pt):
        pts = np.zeros((3, numb_pt))
        theta = np.linspace(0, 2 * pi, numb_pt).reshape(-1, 1)
        pts = (
            self.center
            + self.radius * cos(theta) * self.a
            + self.radius * sin(theta) * self.b
        )
        pts = pts.T
        return pts

    def project_pt_to_img(self, img, intrinsic, numb_pt):
        pts = self.generate_parametric(numb_pt)
        projected = intrinsic @ pts
        projected[0, :] = projected[0, :] / projected[2, :]
        projected[1, :] = projected[1, :] / projected[2, :]
        projected[2, :] = projected[2, :] / projected[2, :]

        # img = np.zeros((480, 640, 3))
        for xp, yp in zip(projected[0, :], projected[1, :]):
            img = cv2.circle(
                img, (int(xp), int(yp)), radius=1, color=(0, 255, 0), thickness=-1
            )

        return img


if __name__ == "__main__":
    rospy.init_node("image_listener")
    saver = ImageSaver()
    img = saver.left_frame

    # Ground truth
    radius = 0.1018

    # Read camera parameters
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

    focal_length = (mtx[0, 0] + mtx[1, 1]) / 2

    # Normalize ellipse coefficients
    ellipse = ellipse_2d()
    ellipse.load_coeff("./juan-scripts/output/ellipse_coefficients.txt")
    c_mat = ellipse.get_c_matrix()

    c_mat[0, 0] = c_mat[0, 0]
    c_mat[0, 1] = c_mat[0, 1]
    c_mat[0, 2] = c_mat[0, 2] / (focal_length)
    c_mat[1, 0] = c_mat[1, 0]
    c_mat[1, 1] = c_mat[1, 1]
    c_mat[1, 2] = c_mat[1, 2] / (focal_length)
    c_mat[2, 0] = c_mat[2, 0] / (focal_length)
    c_mat[2, 1] = c_mat[2, 1] / (focal_length)
    c_mat[2, 2] = c_mat[2, 2] / (focal_length * focal_length)

    # Calculate Eigen vectors
    ret, W, V = cv2.eigen(c_mat)
    V = V.transpose()

    e1 = W[0, 0]
    e2 = W[1, 0]
    e3 = W[2, 0]

    S1 = [+1, +1, +1, +1, -1, -1, -1, -1]
    S2 = [+1, +1, -1, -1, +1, +1, -1, -1]
    S3 = [+1, -1, +1, -1, +1, -1, +1, -1]

    g = m.sqrt((e2 - e3) / (e1 - e3))
    h = m.sqrt((e1 - e2) / (e1 - e3))

    # Calculate two possible solutions
    translations = np.zeros((3, 2))
    normals = np.zeros((3, 2))
    projections = np.zeros((2, 2))

    k = 0
    for i in range(8):
        z0 = S3[i] * (e2 * radius) / m.sqrt(-e1 * e3)
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
            pc = mtx @ t
            projections[:, k] = [pc[0] / pc[2], pc[1] / pc[2]]
            k += 1

    print("algorithm summary")
    print("camera_matrix")
    print(mtx)
    print("focal length {:0.4f}".format(focal_length))
    print("ellipse c matrix")
    print(c_mat)
    print("eigen values")
    print(W)
    print("eigen vectors")
    print(V)

    circles = []
    for k in range(2):
        circles.append(Circle_3d(translations[:, k], normals[:, k], radius, mtx))
        print("solution {:d}".format(k))
        print("pose")
        print(translations[:, k])
        print("normal")
        print(normals[:, k])
        print("projections")
        print(projections[:, k])

    # Draw the ellipse
    df = pd.read_csv("./juan-scripts/output/sample_ellipse_01.txt")
    X = df["x"].values.reshape(-1, 1)
    Y = df["y"].values.reshape(-1, 1)

    for i in range(2):
        # img = np.zeros((480, 640, 3))
        img = saver.left_frame

        # Sample 3D circle
        pts = circles[i].generate_parametric(30)
        df = pd.DataFrame(pts.T, columns=["x", "y", "z"])
        df.to_csv("circle{:d}.txt".format(i), index=None)

        # Draw 3D circle
        img = circles[i].project_pt_to_img(img, mtx, 30)

        # #Draw ellipse samples
        for xp, yp in zip(X.squeeze(), Y.squeeze()):
            img = cv2.circle(
                img, (int(xp), int(yp)), radius=2, color=(0, 0, 255), thickness=-1
            )

        cv2.imshow("img", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows
