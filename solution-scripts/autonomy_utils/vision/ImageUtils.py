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
