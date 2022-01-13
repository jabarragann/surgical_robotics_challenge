import cv2 as cv
import numpy as np


def segment_needle(frame):
    # Convert white pixels into black pixels
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    frame[thresh == 255] = 0

    # Threshold the image
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    # frame_threshold = cv.inRange(frame_HSV, (0,84,72), (180,254,197))
    frame_threshold = cv.inRange(frame_HSV, (0, 0, 27), (180, 84, 255))
    # Dilatation et erosion
    kernel = np.ones((15, 15), np.uint8)
    img_dilation = cv.dilate(frame_threshold, kernel, iterations=1)
    img_erode = cv.erode(img_dilation, kernel, iterations=1)
    # clean all noise after dilatation and erosion
    img_erode = cv.medianBlur(img_erode, 7)
    # Back to RGB
    img_erode = cv.cvtColor(img_erode, cv.COLOR_GRAY2BGR)

    return img_erode
