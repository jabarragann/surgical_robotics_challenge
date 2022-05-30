import numpy as np
import PyKDL

quaternion = (0.48990610366965426, 0.5098941160547137, 0.5098941160547138, -0.48990610366965437)
quaternion = (0.48990610366965426, 0.5098941160547137, 0.5098941160547138, -0.48990610366965437)


class CameraModel:
    fvg = 1.2
    width = 1920
    height = 1080
    cx = width / 2
    cy = height / 2

    f = height / (2 * np.tan(fvg / 2))

    intrinsic_params = np.zeros((3, 3))
    intrinsic_params[0, 0] = f
    intrinsic_params[1, 1] = f
    intrinsic_params[0, 2] = width / 2
    intrinsic_params[1, 2] = height / 2
    intrinsic_params[2, 2] = 1.0
    mtx = intrinsic_params

    focal_length = (mtx[0, 0] + mtx[1, 1]) / 2
