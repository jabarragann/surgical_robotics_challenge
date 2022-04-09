import cv2 as cv
import numpy as np
from autonomy_utils.ambf_utils import AMBFStereoRig, AMBFCamera, AMBFNeedle
from time import time


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


class NeedleSegmenter:
    def __init__(self, ambf_client=None, log=None, needle_handle=None, stereo_rig_handle=None) -> None:
        if needle_handle is not None and stereo_rig_handle is not None:
            self.needle_handle = needle_handle
            self.stereo_rig_handle = stereo_rig_handle
        else:
            self.needle_handle = AMBFNeedle(ambf_client=ambf_client, logger=log)
            self.stereo_rig_handle = AMBFStereoRig(ambf_client=ambf_client)

    @classmethod
    def from_handler(cls, needle_handle, stereo_rig_handle):
        return NeedleSegmenter(needle_handle=needle_handle, stereo_rig_handle=stereo_rig_handle)

    def obtain_needle_pt(self, camera_selector: str):

        needle_pts = self.needle_handle.sample_3d_pts(10)
        T_CN = self.needle_handle.get_needle_to_camera_pose(camera_selector)
        img_pt = self.stereo_rig_handle.project_points(camera_selector, T_CN, needle_pts)

        return img_pt

    # @timer_func
    def clean_image(self, frame: np.ndarray, camera_selector: str, ambf_client=None, log=None) -> np.ndarray:
        """Clean segmentation mask using the location of the needle w.r.t world coordinates.

        Args:
            frame (np.ndarray): [description]
            camera_selector (str): [description]
            ambf_client ([type], optional): [description]. Defaults to None.
            log ([type], optional): [description]. Defaults to None.

        Returns:
            np.ndarray: Clean image
        """
        img_pt = self.obtain_needle_pt(camera_selector)

        # Create mask around needle_pt
        centroid = img_pt.mean(axis=0).squeeze()

        # calculate radius
        radius = 0
        for i in range(img_pt.shape[0]):
            norm = np.linalg.norm(img_pt[i].squeeze() - centroid)
            if norm > radius:
                radius = norm

        # Create a white blob with centroid and radius
        centroid = centroid.astype(int).squeeze().tolist()
        radius = 1.2 * radius
        mask = np.zeros((1080, 1920), dtype="uint8")
        mask = cv.circle(mask, tuple(centroid), int(radius), (255, 255, 255), -1)

        # cv.imshow("mask", mask)
        # cv.waitKey(0)
        # Display image
        # min_x, max_x = 2000, 0
        # min_y, max_y = 2000, 0
        # for i in range(img_pt.shape[0]):
        #     if img_pt[i, 0, 0] < min_x:
        #         min_x = img_pt[i, 0, 0]
        #     if img_pt[i, 0, 0] > max_x:
        #         max_x = img_pt[i, 0, 0]
        #     if img_pt[i, 0, 1] < min_y:
        #         min_y = img_pt[i, 0, 1]
        #     if img_pt[i, 0, 1] > max_y:
        #         max_y = img_pt[i, 0, 1]
        #     # frame = cv2.circle(frame, (int(img_pt[i, 0, 0]), int(img_pt[i, 0, 1])), 5, (255, 0, 0), -1)

        # mask = np.zeros((1080, 1920), dtype="uint8")
        # shift = 60
        # mask = cv.rectangle(
        #     mask,
        #     (int(min_x - shift), int(min_y - shift)),
        #     (int(max_x + shift), int(max_y + shift)),
        #     (255, 0, 0),
        #     -1,
        # )
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray = cv.bitwise_and(gray, mask)
        frame = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        # cv.imshow("frame", frame)
        # cv.waitKey(0)

        return frame

    @staticmethod
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
