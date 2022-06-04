from pathlib import Path
from tkinter import Image
from autonomy_utils.DeepModels.InferencePipeline import InferencePipe
from autonomy_utils.DeepModels.Dice import DiceLoss, DiceScore, DiceBCELoss
from autonomy_utils.ambf_utils import ImageSaver
import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import cv2
import time
import random

import rospy


class ImageProcessing:
    def calculate_needle_salient_points(img: np.ndarray):
        """Calculate medial axis of the needle and tip and tail location. Each point is represented with its x and y
        coordinates.

        Parameters
        ----------
        imag : np.ndarray
            _description_

        Returns
        -------
        medial_axis: np.ndarray
            Medial axis of the needle
        endpts: np.ndarray
            Tip and tail of the needle
        contour:
            Needle contour
        bb:
            Opencv bounding box. Tuple containing x,y,w,h.
        """

        # Find contour
        cnt = ImageProcessing.find_contour(img)

        # fill out holes in the needle
        # Solution from https://stackoverflow.com/questions/10316057/filling-holes-inside-a-binary-object
        # img2 = np.copy(img)
        # cv2.drawContours(img, [cnt], 0, (255, 255, 255), -1)

        # Remove sharp edges from cropped image
        # kernel = np.ones((5, 5), np.uint8)
        # for i in range(1):
        #     img = cv2.erode(img, kernel, iterations=2)
        #     img = cv2.dilate(img, kernel, iterations=1)

        x, y, w, h = cv2.boundingRect(cnt)
        p = 15
        x, y = x - p, y - p
        w, h = w + p, h + p
        crop_img = img[y : y + h, x : x + w]
        bb = (x, y, w, h)

        # Calculate on cropped image
        medial_axis, skel = ImageProcessing.calculate_medial_axis(crop_img)
        endpts = ImageProcessing.calculate_medial_endpoints(skel)
        # Go back to full resolution
        offset = np.array([y, x]).reshape((1, 2))
        medial_axis = medial_axis + offset
        endpts = endpts + offset

        # Change to x,y representation
        medial_axis[:, :] = medial_axis[:, [1, 0]]
        endpts[:, :] = endpts[:, [1, 0]]

        if endpts.shape[0] > 2:
            raise Exception("ImageUtils.NeedleSalientPointDetector generated more than 2 points for tip and tail.")

        return medial_axis.tolist(), endpts.tolist(), cnt, bb

    def find_contour(img: np.ndarray):
        """Find the biggest contour in an image

        Parameters
        ----------
        img : np.ndarray
            RGB image

        Returns
        -------
        contour:
            Biggest contour in the image.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        cnts, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get only the biggest contour
        max_area = 0
        max_cnt = 0
        for c in cnts:
            area = cv2.contourArea(c)
            if area > max_area:
                max_cnt = c
                max_are = area

        return max_cnt

    def calculate_medial_axis(segmented_img):
        """_summary_

        Parameters
        ----------
        image : _type_
            _description_
        """
        if len(segmented_img.shape) == 3:
            data = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2GRAY)
        else:
            data = segmented_img

        binary = data > filters.threshold_otsu(data)
        # do the skeletonization
        skel = morphology.skeletonize(binary)
        skel = (skel * 255).astype(np.uint8)
        pt_along_axis = np.argwhere(skel > 200)
        return pt_along_axis, skel

    def calculate_medial_endpoints(medial_axis: np.ndarray):
        """_summary_

        Parameters
        ----------
        medial_axis : np.ndarray
            _description_
        """

        def lineEnds(P):
            """find the Central pixel and one ajacent pixel is said to be a line start or line end"""
            return 255 * ((P[4] == 255) and np.sum(P) == 510)

        tip_tail_result = generic_filter(medial_axis, lineEnds, (3, 3))
        tip_tail_result = np.argwhere(tip_tail_result > 200)

        return tip_tail_result


if __name__ == "__main__":

    rospy.init_node("image_listener")
    img_saver = ImageSaver()
    time.sleep(0.3)

    model_path = Path("./Resources/segmentation_weights/best_model_512.pth")
    inference_model = InferencePipe(model_path, device="cuda")
    segmented_l_raw = img_saver.get_current_frame("right")
    segmented_l_raw = inference_model.segmented_image(segmented_l_raw)
    segmented_l_raw = cv2.cvtColor(segmented_l_raw, cv2.COLOR_GRAY2BGR)

    # segmented_l_raw = cv2.imread("./Media/test_img/problematic_segmentation.jpeg")
    # segmented_l_raw = cv2.cvtColor(segmented_l_raw, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("./Media/test_img/problematic_segmentation.jpeg", segmented_l_raw)
    # segmented_l_raw = cv2.imread("./Media/test_img/segmented_needle01.jpeg")
    # segmented_l_raw = cv2.imread("to_erase/20220113151427_l_seg.jpeg")

    print(segmented_l_raw.shape)
    medial_axis, endpts, cnt, bb = ImageProcessing.calculate_needle_salient_points(segmented_l_raw)
    print(endpts)

    for pt in medial_axis:
        segmented_l_raw[pt[1], pt[0], :] = [0, 0, 255]
    for pt in endpts:
        segmented_l_raw[pt[1], pt[0], :] = [0, 0, 0]

    x, y, w, h = bb
    segmented_l_cropped = np.copy(segmented_l_raw[y : y + h, x : x + w])
    # segmented_l_raw = cv2.rectangle(segmented_l_raw, (x, y), (x + w, y + h), (255, 0, 255), 2)
    # segmented_l_raw = cv2.rectangle(np.copy(segmented_l_raw), (x, y), (x + w, y + h), (255, 0, 0), 2)

    w_name = "final"
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)
    cv2.imshow(w_name, segmented_l_cropped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.imshow(w_name, segmented_l_raw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
