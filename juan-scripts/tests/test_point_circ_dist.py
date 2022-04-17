"""
Test taken from 
https://www.geometrictools.com/Documentation/DistanceToCircle3.pdf
"""
import numpy as np
from autonomy_utils.circle_pose_estimator import Circle3D
from autonomy_utils import Logger
import rospy
from spatialmath.base import q2r

log = Logger.Logger(name="test").log
np.set_printoptions(precision=4, suppress=True, sign=" ")

if __name__ == "__main__":

    small_circle = Circle3D(np.array([2, 2, 0]), np.array([0, 0, 1]), 1)
    large_circle = Circle3D(np.array([2, 2, 0]), np.array([0, 0, 1]), 2)

    log.info("Test1")
    test_pt1 = np.array([2, 6, 0])
    closest_pt1 = large_circle.closest_pt_in_circ_to_pt(test_pt1)
    log.info(f"Closest to large {closest_pt1}")  # Closest to large: [2,4,0]
    log.info(f"Distance to large {np.linalg.norm(test_pt1-closest_pt1)}")  # Distance to large: 2
    closest_pt1 = small_circle.closest_pt_in_circ_to_pt(test_pt1)
    log.info(f"Closest to small {closest_pt1}")  # Closest to small: [2,3,0]
    log.info(f"Distance to large {np.linalg.norm(test_pt1-closest_pt1)}")  # Distance to small: 3

    log.info("Test2")
    test_pt2 = np.array([0.5, 2, 4])
    closest_pt2 = large_circle.closest_pt_in_circ_to_pt(test_pt2)
    log.info(f"Closest to large {closest_pt2}")  # Closest to large: [0,2,0]
    log.info(f"Distance to large {np.linalg.norm(test_pt2-closest_pt2)}")  # Distance to large: 2
    closest_pt2 = small_circle.closest_pt_in_circ_to_pt(test_pt2)
    log.info(f"Closest to small {closest_pt2}")  # Closest to small: [1,2,0]
    log.info(f"Distance to large {np.linalg.norm(test_pt2-closest_pt2)}")  # Distance to small: 3

    log.info("Apply random rigid transformation")

    rotation = np.array([1, 2, 3, 4])
    rotation = rotation / np.linalg.norm(rotation)
    rotation = q2r(rotation)
    translation = np.array([0.1234, 5.6789, -1.9735])

    log.info(f"random rotation \n{rotation}")
    log.info(f"random translation\n{translation}")

    center = rotation @ np.array([2, 2, 0]) + translation
    normal = rotation @ np.array([0, 0, 1])
    large_circle = Circle3D(center, normal, 2)
    test_pt1 = rotation @ np.array([2, 6, 0]) + translation
    test_pt2 = rotation @ np.array([0.5, 2, 4]) + translation

    closest_pt1 = large_circle.closest_pt_in_circ_to_pt(test_pt1)
    log.info(f"Distance to large transf {np.linalg.norm(test_pt1-closest_pt1)}")  # Distance to large: 2
    assert np.isclose(np.linalg.norm(test_pt1 - closest_pt1), 2.0), "Function failed"

    closest_pt2 = large_circle.closest_pt_in_circ_to_pt(test_pt2)
    log.info(f"Distance to large transf {np.linalg.norm(test_pt2-closest_pt2)}")  # Distance to large: 2
    assert np.isclose(np.linalg.norm(test_pt2 - closest_pt2), 4.03112887414), "Function failed"
