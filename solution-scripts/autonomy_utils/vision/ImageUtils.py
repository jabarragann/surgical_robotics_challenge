import numpy as np
from scipy import ndimage as ndi
from skimage import morphology, filters
from scipy.ndimage import generic_filter
import matplotlib.pyplot as plt
import cv2
import time
import random


def lineEnds(P):
    """find the Central pixel and one ajacent pixel is said to be a line start or line end"""
    return 255 * ((P[4] == 255) and np.sum(P) == 510)


# Compute the medial axis (skeleton)
def locate_points(img: np.ndarray, pt_along_needle=20):
    # convert to greyscale
    data = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = data > filters.threshold_otsu(data)
    # do the skeletonization
    skel = morphology.skeletonize(binary)
    find = skel.copy()
    points_along = skel.copy() * 255
    # locate the tip/tail using filters
    result = generic_filter(skel * 255, lineEnds, (3, 3))
    print(np.sum(result))
    x, y = find.shape
    points = []
    res = []
    for i in range(x):
        for j in range(y):
            if points_along[i, j] == 255:
                points.append((j, i))
            if result[i, j] == 255:
                res.append((j, i))
    points_along_needle = random.sample(points, pt_along_needle)
    return data, res, points_along_needle


def locate_points2(img: np.ndarray, pt_along_needle=20):
    scale_percent = 20  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # convert to greyscale
    data = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

    binary = data > filters.threshold_otsu(data)
    # do the skeletonization
    skel = morphology.skeletonize(binary)
    find = skel.copy()
    points_along = skel.copy() * 255

    # locate the tip/tail using filters
    result = generic_filter(skel * 255, lineEnds, (3, 3))
    print(np.sum(result))
    x, y = find.shape
    points = []
    res = []
    for i in range(x):
        for j in range(y):
            if points_along[i, j] == 255:
                points.append((int(j / 0.2), int(i / 0.2)))
            if result[i, j] == 255:
                res.append((int(j / 0.2), int(i / 0.2)))
    points_along_needle = random.sample(points, pt_along_needle)
    return data, res, points_along_needle


class SalientPtLocator:
    def __init__(self):
        self.tip_tail_pt = None
        self.pt_along_axis = None

    def locate_points2(self, img: np.ndarray, pt_along_needle=20):
        scale_percent = 20  # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)

        # resize image
        resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        # convert to greyscale
        data = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

        binary = data > filters.threshold_otsu(data)
        # do the skeletonization
        skel = morphology.skeletonize(binary)
        find = skel.copy()
        points_along = skel.copy() * 255

        # cv2.imshow("img", points_along.astype(np.int8))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # locate the tip/tail using filters
        result = generic_filter(skel * 255, lineEnds, (3, 3))
        print(np.sum(result))
        x, y = find.shape
        points = []
        res = []
        for i in range(x):
            for j in range(y):
                if points_along[i, j] == 255:
                    points.append((int(j / 0.2), int(i / 0.2)))
                if result[i, j] == 255:
                    res.append((int(j / 0.2), int(i / 0.2)))

        self.tip_tail_pt = res
        self.points_along_needle = np.array(random.sample(points, pt_along_needle))

        return data, res, self.points_along_needle

    def draw_solution(self, segmented_l):
        segmented_l = cv2.circle(segmented_l, self.tip_tail_pt[0], 10, (255, 0, 0), -1)
        segmented_l = cv2.circle(segmented_l, self.tip_tail_pt[1], 10, (255, 0, 0), -1)

        for i in range(self.points_along_needle.shape[0]):
            segmented_l = cv2.circle(segmented_l, self.points_along_needle[i, :], 6, (0, 255, 0), -1)
        return segmented_l
