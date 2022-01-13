from re import I
from autonomy_utils.ambf_utils import AMBFStereoRig, ImageSaver, AMBFCamera, AMBFNeedle
from autonomy_utils import image_segmentation, Logger
import cv2
import rospy
import numpy as np
import ambf_client as Client
from ambf_client import Client
from surgical_robotics_challenge.camera import Camera
import tf_conversions.posemath as pm
from pathlib import Path
import datetime

np.set_printoptions(precision=3, suppress=True)


def clean_image(frame, camera_selector):
    # Paint needle points
    needle_handle = AMBFNeedle(ambf_client=c, logger=log)
    camera_handle = AMBFCamera(ambf_client=c, camera_selector=camera_selector)
    stereo_rig_handle = AMBFStereoRig(ambf_client=c)
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
    mask = cv2.rectangle(
        mask,
        (int(min_x - shift), int(min_y - shift)),
        (int(max_x + shift), int(max_y + shift)),
        (255, 0, 0),
        -1,
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_and(gray, mask)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame


def save_image(path, l_img, l_seg, r_img, r_seg):
    # generate timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(str(path / f"{ts}_l_img.jpeg"), l_img)
    cv2.imwrite(str(path / f"{ts}_r_img.jpeg"), r_img)
    cv2.imwrite(str(path / f"{ts}_l_seg.jpeg"), l_seg)
    cv2.imwrite(str(path / f"{ts}_r_seg.jpeg"), r_seg)
    print("images saved ...")


def main():
    image_subs = ImageSaver()
    w_name = "disp_final"
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)

    p = Path("/home/juan1995/research_juan/needle_dataset_/d2")
    if not p.exists():
        p.mkdir(parents=True)

    camera_selector = "left"

    while True:
        frame_l = image_subs.get_current_frame("left")
        frame_r = image_subs.get_current_frame("right")
        segmented_l = image_segmentation.segment_needle(frame_l)
        segmented_r = image_segmentation.segment_needle(frame_r)

        final = np.hstack((frame_l, segmented_l))
        # log.info(f"T_CN \n {T_CN}")
        # log.info(f"T_WF \n {stereo_rig_handle.get_current_pose()}")
        # stereo_rig = Camera(c, "CameraFrame")
        # log.info(f"T_WF \n {pm.toMatrix(stereo_rig.get_T_c_w())}")

        cv2.imshow(w_name, final)
        k = cv2.waitKey(3) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("s"):
            clean_seg_l = clean_image(segmented_l, "left")
            clean_seg_r = clean_image(segmented_r, "right")

            final_2 = np.hstack((clean_seg_l, clean_seg_r))
            save_image(p, frame_l, clean_seg_l, frame_r, clean_seg_r)

            # cv2.namedWindow("saved", cv2.WINDOW_NORMAL)
            # cv2.imshow("saved", final_2)
            # cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    rospy.init_node("dataset_collection")
    c = Client("juanclient")
    log = Logger.Logger().log
    c.connect()

    main()