from pathlib import Path
import time
from autonomy_utils.DeepModels.InferencePipeline import InferencePipe
from autonomy_utils.DeepModels.Dice import DiceLoss, DiceScore, DiceBCELoss
from autonomy_utils.Logger import Logger
from autonomy_utils.NeedlePoseConstructor import NeedlePoseConstructor
from autonomy_utils.ambf_utils import AMBFNeedle, ImageSaver
from autonomy_utils.circle_pose_estimator import CirclePoseEstimator, Ellipse2D
from autonomy_utils.utils.Utils import find_correspondent_pt
from autonomy_utils.vision.ImageUtils import ImageProcessing
from autonomy_utils.Models.CameraModel import CameraModel
from autonomy_utils.Models.NeedleModel import NeedleModel
from autonomy_utils.vision.StereoUtils import StereoLocator
from ambf_client import Client

import numpy as np
import rospy

log = Logger("Complete pipeline").log


class NeedlePoseEstimator:
    """Needle pose estimator class"""

    def __init__(self, model_path, ambf_client, device: str = "cuda") -> None:

        if not model_path.exists():
            log.error("Segmentation model weights not found")
            exit(0)

        self.inference_model = InferencePipe(model_path, device=device)
        self.stereo_locator = StereoLocator(ambf_client)

    def estimate_pose(self, left_img, right_img):

        # ------------------------------------------------------------
        # Segmented images
        # ------------------------------------------------------------
        segmented_l = self.inference_model.segmented_image(left_img)
        segmented_r = self.inference_model.segmented_image(right_img)

        # ------------------------------------------------------------
        # Extract image features
        # ------------------------------------------------------------
        medial_ax_l, tip_tail_pix_l, _, bb = ImageProcessing.calculate_needle_salient_points(segmented_l)
        medial_ax_r, tip_tail_pix_r, _, bb = ImageProcessing.calculate_needle_salient_points(segmented_r)

        # ------------------------------------------------------------
        # Stereo calculations
        # ------------------------------------------------------------
        tip_tail_3d_est = self.stereo_locator.locate_tip_tail_3d(tip_tail_pix_l, tip_tail_pix_r)

        # ------------------------------------------------------------
        # Circle estimation
        # ------------------------------------------------------------
        X = np.array(medial_ax_l)[:, 0].reshape(-1, 1)
        Y = np.array(medial_ax_l)[:, 1].reshape(-1, 1)
        ellipse = Ellipse2D.from_sample_points_skimage(X - CameraModel.cx, Y - CameraModel.cy)
        estimator = CirclePoseEstimator(ellipse, CameraModel.mtx, CameraModel.focal_length, NeedleModel.radius)
        circles = estimator.estimate_pose()

        # ------------------------------------------------------------
        # Pose construction
        # ------------------------------------------------------------
        needle_pose_constructor = NeedlePoseConstructor(circles, tip_tail_3d_est)
        needle_pose_est = needle_pose_constructor.pose

        return needle_pose_est, tip_tail_3d_est


if __name__ == "__main__":

    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    c = Client("juanclient")
    c.connect()
    time.sleep(0.3)

    model_path = Path("./Resources/segmentation_weights/best_model_512.pth")
    needle_pose_estimator = NeedlePoseEstimator(model_path, c)

    left_img = img_saver.get_current_frame("left")
    right_img = img_saver.get_current_frame("right")

    pose_est, tip_tail_3d_est = needle_pose_estimator.estimate_pose(left_img, right_img)

    log.info(f"Needle pose detected \n{pose_est} ")
    log.info(f"Tip tail 3d position \n{tip_tail_3d_est} ")
    log.info(f"\n\n")

    # ------------------------------------------------------------
    # Evaluate solution
    # ------------------------------------------------------------
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_selector = "left"
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    tip_tail_pt = T_CN @ needle_handle.get_tip_tail_pose().T

    tip_tail_3d_est, tip_tail_pt = find_correspondent_pt(tip_tail_3d_est, tip_tail_pt)

    log.info("*" * 30)
    log.info("Triangulation evaluation")
    log.info("*" * 30)
    log.info(f"Error (mm) {1000*np.linalg.norm(tip_tail_3d_est-tip_tail_pt,axis=0)}")
    log.info(f"Error (mm) {1000*np.linalg.norm(tip_tail_3d_est-tip_tail_pt,axis=0).mean():0.05f}")
    log.info(f"\n\n")

    log.info("*" * 30)
    log.info("Needle pose estimate evaluation")
    log.info("*" * 30)

    needle_handle.pose_estimate_evaluation(pose_est, camera_selector)
