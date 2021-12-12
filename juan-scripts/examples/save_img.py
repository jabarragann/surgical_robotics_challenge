from re import I
from autonomy_utils.ambf_utils import ImageSaver
from pathlib import Path
import time
import rospy

if __name__ == "__main__":

    rospy.init_node("image_listener")
    saver = ImageSaver()
    p = Path(__file__).parent / "img"
    saver.save_frame("left", p)
