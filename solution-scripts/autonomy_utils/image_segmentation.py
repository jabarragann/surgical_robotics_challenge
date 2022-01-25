import cv2 as cv
import numpy as np
from autonomy_utils.ambf_utils import AMBFStereoRig, AMBFCamera, AMBFNeedle


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


def clean_image(frame: np.ndarray, camera_selector: str, ambf_client=None, log=None) -> np.ndarray:
    """Clean segmentation mask using the location of the needle w.r.t world coordinates.

    Args:
        frame (np.ndarray): [description]
        camera_selector (str): [description]
        ambf_client ([type], optional): [description]. Defaults to None.
        log ([type], optional): [description]. Defaults to None.

    Returns:
        np.ndarray: Clean image
    """
    # Paint needle points
    needle_handle = AMBFNeedle(ambf_client=ambf_client, logger=log)
    camera_handle = AMBFCamera(ambf_client=ambf_client, camera_selector=camera_selector)
    stereo_rig_handle = AMBFStereoRig(ambf_client=ambf_client)
    needle_pts = needle_handle.sample_3d_pts(10)
    T_CN = needle_handle.get_needle_to_camera_pose(camera_selector)
    img_pt = camera_handle.project_points(T_CN, needle_pts)

    # Display image
    min_x, max_x = 2000, 0
    min_y, max_y = 2000, 0
    for i in range(img_pt.shape[0]):
        if img_pt[i, 0, 0] < min_x:
            min_x = img_pt[i, 0, 0]
        if img_pt[i, 0, 0] > max_x:
            max_x = img_pt[i, 0, 0]
        if img_pt[i, 0, 1] < min_y:
            min_y = img_pt[i, 0, 1]
        if img_pt[i, 0, 1] > max_y:
            max_y = img_pt[i, 0, 1]
        # frame = cv2.circle(frame, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 5, (255, 0, 0), -1)

    mask = np.zeros((1080, 1920), dtype="uint8")
    shift = 60
    mask = cv.rectangle(
        mask,
        (int(min_x - shift), int(min_y - shift)),
        (int(max_x + shift), int(max_y + shift)),
        (255, 0, 0),
        -1,
    )
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.bitwise_and(gray, mask)
    frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    return frame
