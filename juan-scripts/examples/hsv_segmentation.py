"""
Script to create a files with all the points on top of the needle
Logic
(1) Select a roi where the needle is located
(2) Save all the points in the ROI that correspond to the needle
(3) Show selected points in green
"""

import cv2
from pathlib import Path
import numpy as np
import pandas as pd

mouseX, mouseY = None, None


def get_pixel_values(event, x, y, flags, param):
    global mouseX, mouseY, final
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        print(mouseX, mouseY)
        print(final[mouseY, mouseX])


def save_pts_(file_path: Path, img_pt):
    results_df = pd.DataFrame(columns=["id", "x", "y"])
    for i in range(img_pt.shape[0]):
        # Save pts
        results_df = results_df.append(
            pd.DataFrame(
                [[i, int(img_pt[i, 0]), int(img_pt[i, 1])]],
                columns=["id", "x", "y"],
            )
        )
    path = Path(file_path)
    if not path.parent.exists():
        path.parent.mkdir()

    results_df.to_csv(file_path, index=None)


if __name__ == "__main__":

    path = Path(__file__).parent

    img = cv2.imread(str(path / "img/left_frame.jpeg"))

    # Get roi
    roi = cv2.selectROI(img)
    print(roi)
    roi_cropped = img[int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])]
    # See image in hsv format
    hsv_roi = cv2.cvtColor(roi_cropped, cv2.COLOR_BGR2HSV)
    final = np.hstack((hsv_roi[:, :, 0], hsv_roi[:, :, 1], hsv_roi[:, :, 2]))
    # Threshold the whole image
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    (T, thresh_img) = cv2.threshold(hsv_img[:, :, 1], 60, 255, cv2.THRESH_BINARY)
    # Get pixels correspoind to the needle
    needle_pts = []
    for i in range(roi[1], roi[1] + roi[3]):
        for j in range(roi[0], roi[0] + roi[2]):
            if thresh_img[i, j] == 0:
                needle_pts.append([j, i])  # Save x,y positions. Origin in the left top corner
                img[i, j] = [0, 255, 0]
    needle_pts = np.array(needle_pts)
    # Save pixels locations
    save_pts_(path / "needle_segmentation_pts.txt", needle_pts)

    # Show selected points
    w_name = "selected pts"
    cv2.imshow(w_name, img)
    cv2.setMouseCallback(w_name, get_pixel_values)
    cv2.waitKey(0)
    cv2.destroyAllWindows()