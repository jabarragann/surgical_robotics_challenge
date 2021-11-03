"""
Script to save a single frame with the corresponding pose of the camera and needle.

"""

from surgical_robotics_challenge.scene import Scene 
from surgical_robotics_challenge.camera import Camera 
from ambf_client import Client

import time
import numpy as np 
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import json
import tf_conversions.posemath as pm

class ImageSaver:

    def __init__(self):
        self.bridge = CvBridge()

        self.img_subs = rospy.Subscriber("/ambf/env/cameras/cameraL/ImageData", Image, self.left_callback)
        
        self.left_frame = None
        self.left_ts = None
        
        #AMBF client
        self.scene = None 
        self.ambf_cam = None
        self.init_ambf_client() #Init scene and ambf_cam

        # Wait a until subscribers and publishers are ready
        rospy.sleep(0.5)
    
    def init_ambf_client(self):
        c = Client()
        c.connect()
        self.scene = Scene(c)
        time.sleep(0.3)
        
        self.ambf_cam = Camera(c,"cameraL")

    def save_frame(self):
        # Save frame
        cv2.imwrite('./data/'+str(self.left_ts)+'.jpeg', self.left_frame)

        # Save spatial information
        json_data = self.get_spatial_info()

        with open('./data/'+str(self.left_ts)+'.txt', 'w') as outfile:
            json.dump(json_data, outfile,indent=4)

    def left_callback(self, msg):
        try:
            cv2_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.left_frame = cv2_img
            self.left_ts = msg.header.stamp
        except CvBridgeError, e:
            print(e)
    
    def get_spatial_info(self):
        cam_pose = pm.toMatrix(self.ambf_cam.get_T_w_c())
        needle_pose = pm.toMatrix(self.scene.needle_measured_cp())
        
        data = {}
        data['needle'] = {
            'name': 'needle',
            'rotation': needle_pose[:3,:3].tolist(),
            'position': needle_pose[:3,3].tolist(),
            'pose': needle_pose.tolist()
        }
        data['camera'] = {
            'name': 'left_camera',
            'rotation': cam_pose[:3,:3].tolist(),
            'position': cam_pose[:3,:3].tolist(),
            'pose': cam_pose.tolist()
        }
            
            
        return data

def main():
    rospy.init_node('image_listener')
    
    save = ImageSaver()
    
    save.save_frame()

if __name__ == '__main__':
    main()
