from typing import List
from autonomy_utils.Logger import Logger
from autonomy_utils.circle_pose_estimator import Circle3D
import numpy as np
from spatialmath.base import trnorm
from autonomy_utils.utils.Utils import find_correspondent_pt
from autonomy_utils.NeedleModel import NeedleModel

log = Logger("Pose constructor").log

needle_model = NeedleModel()


class NeedlePoseConstructor:
    def __init__(self, circles: List[Circle3D], tip_tail_W: np.ndarray) -> None:
        best_idx = self.choose_best_circle(circles, tip_tail_W)

        # To identify the needle's pose you need to know the location of tail.
        # Since, I don't know this information I will create to different transformations
        # and select the one that produces the least error.

        self.pose_0 = self.circle2needlepose(circles[best_idx], tip_tail_W[:3, 0])
        self.pose_1 = self.circle2needlepose(circles[best_idx], tip_tail_W[:3, 1])

        # Get location of tip/tail in Needle local frame.
        tip_tail_N = needle_model.get_tip_tail_pose()

        error = []
        for k in range(2):
            T_W_N = getattr(self, f"pose_{k}")
            tip_tail_W_est = T_W_N @ tip_tail_N.T
            tip_tail_W_est, tip_tail_W = find_correspondent_pt(tip_tail_W_est, tip_tail_W)
            error.append(np.linalg.norm(tip_tail_W_est - tip_tail_W))

        tmp = min(error)
        best_pose_idx = error.index(tmp)
        self.pose = getattr(self, f"pose_{best_pose_idx}")

    def choose_best_circle(self, circles: List[Circle3D], tip_tail_loc: np.ndarray):
        # Select circle that minimizes distance to triangulated points.
        dist_list = []
        for k in range(2):
            t_list = []
            for i in range(2):
                closest1 = circles[k].closest_pt_in_circ_to_pt(tip_tail_loc[:3, i])
                dist1 = np.linalg.norm(tip_tail_loc[:3, i] - closest1)
                t_list.append(dist1)
            # log.info(t_list)
            dist_list.append((t_list[0] + t_list[1]) / 2)

        dist_list = np.array(dist_list)
        selected_circle = np.argmin(dist_list)
        # log.info(f"distance to estimated circles {dist_list}")

        return selected_circle

    def circle2needlepose(self, circle: Circle3D, tail: np.ndarray) -> np.ndarray:
        est_center = circle.center
        est_normal = circle.normal
        est_normal = est_normal / np.sqrt(est_normal.dot(est_normal))
        est_x = -(tail - circle.center)
        est_x = est_x / np.linalg.norm(est_x)
        est_y = np.cross(est_normal, est_x)
        est_y = est_y / np.sqrt(est_y.dot(est_y))

        # Construct matrix
        pose_est = np.identity(4)
        pose_est[:3, 0] = est_x
        pose_est[:3, 1] = est_y
        pose_est[:3, 2] = est_normal
        pose_est[:3, 3] = est_center

        # re orthogonalize rotation matrix
        pose_est[:3, :3] = trnorm(pose_est[:3, :3])

        return pose_est
