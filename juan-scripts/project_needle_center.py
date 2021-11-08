import json
import cv2 
import numpy as np
from numpy.linalg import inv

if __name__ == "__main__":
    # name = "./data/1635920726404870261"
    name = "./data/1636328075966555985"
    fvg = 1.2
    width = 640
    height = 480
    
    f = height/(2*np.tan(fvg/2))

    intrinsic_params = np.zeros((3,3))
    intrinsic_params[0,0] = f
    intrinsic_params[1,1] = f
    intrinsic_params[0,2] = width/2
    intrinsic_params[1,2] = height/2
    intrinsic_params[2,2] = 1.0 


    data = json.load(open(name+'.txt'))
    T_WN = np.zeros((4,4))
    T_WN[:,:] = data['needle']['pose']
    T_WC = np.zeros((4,4))
    T_WC[:,:] = data['camera']['pose']

    T_CN = inv(T_WC).dot(T_WN)
    # T_CN = T_WC.dot(T_WN)

    rvecs,_ = cv2.Rodrigues(T_CN[:3,:3]) 
    tvecs = T_CN[:3,3]

    img_pt, _ = cv2.projectPoints(np.float32([[0,0,0]]), rvecs, tvecs, intrinsic_params, np.float32([0,0,0,0,0]))

    print("intrinsic")
    print(intrinsic_params)
    print("T_WN")
    print(T_WN)
    print("T_WC")
    print(T_WC)
    print("Projected center")
    print(img_pt[0,0])

    

    
    img = cv2.imread(name+'.jpeg')
    img = cv2.circle(img, (int(img_pt[0,0,0]), int(img_pt[0,0,1])), 5, (255,0,0), -1)
    cv2.imshow('img',img) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
