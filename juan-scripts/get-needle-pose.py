from surgical_robotics_challenge.scene import Scene 
from surgical_robotics_challenge.camera import Camera 
from ambf_client import Client
import time
import numpy as np 
import json
import tf_conversions.posemath as pm

if __name__ == "__main__":
    
    c = Client('juanclient')
    c.connect()
    time.sleep(0.3)
    scene = Scene(c)
    time.sleep(0.3)
    
    cam = Camera(c,"cameraL")
    cam_pose = cam.get_T_w_c()
    needle_pose = scene.needle_measured_cp()   

    print("Needle pose w.r.t w")
    print("rotation")
    print(needle_pose.M)
    print("position")
    print(needle_pose.p) 
    print("Camera pose w.r.t w") 
    print("rotation")
    print(cam_pose.M)
    print("position")
    print(cam_pose.p) 
    
    # pose = pm.toMatrix(cam_pose)

    # data = {}
    # data['people'] = []

    # data['people'].append({
    #     'name': 'Larry',
    #     'website': 'google.com',
    #     'from': pose[:3,3].tolist() 
    # })
    # data['people'].append({
    #     'name': 'Tim',
    #     'website': 'apple.com',
    #     'from': 'Alabama'
    # })

    # with open('data.txt', 'w') as outfile:
    #     json.dump(data, outfile)
