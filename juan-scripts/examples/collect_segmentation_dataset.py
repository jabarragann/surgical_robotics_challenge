import os
from re import I
from autonomy_utils.ambf_utils import AMBFStereoRig, ImageSaver, AMBFCamera, AMBFNeedle
from autonomy_utils import Logger
from autonomy_utils.vision.ImageSegmenter import NeedleSegmenter
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


def save_image(path, l_img, l_seg, r_img, r_seg):
    # generate timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    cv2.imwrite(str(path / f"{ts}_l_img.jpeg"), l_img)
    cv2.imwrite(str(path / f"{ts}_r_img.jpeg"), r_img)
    cv2.imwrite(str(path / f"{ts}_l_seg.jpeg"), l_seg)
    cv2.imwrite(str(path / f"{ts}_r_seg.jpeg"), r_seg)


def save_image(path: Path, image: np.ndarray, mask: np.ndarray, next_image: int):
    final_path = path / f"{next_image:04d}"

    if not (final_path / "images").exists():
        (final_path / "images").mkdir(parents=True)
    if not (final_path / "masks").exists():
        (final_path / "masks").mkdir(parents=True)

    cv2.imwrite(str((final_path / "images") / f"{next_image:04d}_image.jpeg"), image)
    cv2.imwrite(str((final_path / "masks") / f"{next_image:04d}_mask.jpeg"), mask)


def main():
    image_subs = ImageSaver()
    w_name = "disp_final"
    cv2.namedWindow(w_name, cv2.WINDOW_NORMAL)

    p = Path("/home/juan1995/research_juan/needle_dataset_/d8")
    if not p.exists():
        p.mkdir(parents=True)

    # Calculate the next number
    sub_dirs = [int(x.name) for x in p.glob("*")]
    if len(sub_dirs) == 0:
        next_img = 0
    else:
        next_img = max(sub_dirs) + 1

    needle_seg = NeedleSegmenter(ambf_client=c, log=log)
    camera_selector = "left"

    count = 0

    print(f"Data collection util cmds")
    print(f"Save next image to {next_img:04d}")
    print(f"Press 's' to save new images")
    print(f"Press 'q' to quit")

    while True:
        frame_l = image_subs.get_current_frame("left")
        frame_r = image_subs.get_current_frame("right")
        segmented_l = NeedleSegmenter.segment_needle(frame_l)
        segmented_r = NeedleSegmenter.segment_needle(frame_r)
        clean_seg_l = needle_seg.clean_image(segmented_l, "left")

        # final = np.hstack((frame_l, segmented_l))
        final = np.hstack((frame_l, clean_seg_l))
        # log.info(f"T_CN \n {T_CN}")
        # log.info(f"T_WF \n {stereo_rig_handle.get_current_pose()}")
        # stereo_rig = Camera(c, "CameraFrame")
        # log.info(f"T_WF \n {pm.toMatrix(stereo_rig.get_T_c_w())}")

        cv2.imshow(w_name, final)
        k = cv2.waitKey(3) & 0xFF

        if k == ord("q"):
            break
        elif k == ord("s"):
            # clean_seg_l = needle_seg.clean_image(segmented_l, "left", ambf_client=c, log=log)
            clean_seg_r = needle_seg.clean_image(segmented_r, "right")

            # final_2 = np.hstack((clean_seg_l, clean_seg_r))
            print(f"saving images {next_img:04d} and  {next_img+1:04d} ...")
            save_image(p, frame_l, clean_seg_l, next_img)
            next_img += 1
            save_image(p, frame_r, clean_seg_r, next_img)
            next_img += 1
            # save_image(p, frame_l, clean_seg_l, frame_r, clean_seg_r)
            count += 1
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
