import cv2
import numpy as np
from pathlib import Path
from numpy.lib.npyio import save
import pandas as pd


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


def run():
    # reading the image
    # img = cv2.imread("./juan-scripts/examples/output/needle_segmentation_pts0.jpeg")
    path = "./output/needle_segmentation_pts0.jpeg"
    img = cv2.imread(path)

    # convert to greyscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # create SIFT feature extractor
    sift = cv2.SIFT_create(nfeatures=18)

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(img, None)

    # convert to point class
    points = cv2.KeyPoint_convert(keypoints)
    points = np.array(points)
    # print(points)
    # Display image
    for i in range(points.shape[0]):
        img = cv2.circle(img, (int(points[i, 0]), int(points[i, 1])), 3, (255, 0, 0), -1)

    # Save points
    dst_path = Path(path).resolve()
    dst_path = dst_path.parent / (dst_path.with_suffix("").name + "_sift")
    dst_path = dst_path.with_suffix(".txt")
    save_pts_(dst_path, points)

    # draw the detected key points
    # sift_image = cv2.drawKeypoints(gray, keypoints, img)
    # show the image
    # cv2.imshow("image", sift_image)
    cv2.imshow("image", img)
    # save the image
    # cv2.imwrite("sift.jpg", sift_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
