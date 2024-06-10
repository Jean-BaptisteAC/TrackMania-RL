import cv2
import numpy as np
import win32.win32gui as wind32
from mss import mss
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error

import os
import sys
sys.path.append('utils')

from dotenv import load_dotenv
load_dotenv()
    

class Image_Vision():
    def __init__(self, dimension_reduction = 4):
        self.hwnd = wind32.FindWindow(None, os.getenv('GAME_WINDOW_NAME'))

        # Crop the screenshot to remove unecessary information and reduce dimensionality
        self.lateral_margin = 10 + 2
        self.upper_margin = 40 + 150
        self.lower_margin = 10 + 50
        self.dimension_reduction = dimension_reduction
            
        self.get_frame()
        self.is_running = True

    def get_frame(self):
        
        left, top, right, bottom = wind32.GetWindowRect(self.hwnd)

        left = left + self.lateral_margin
        right = right - self.lateral_margin
        top = top + self.upper_margin
        bottom = bottom - self.lower_margin

        bounding_box = left, top, right, bottom
        with mss() as sct:
            frame = np.array(sct.grab(bounding_box)) 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dim = (int(frame.shape[1]/self.dimension_reduction),
               int(frame.shape[0]/self.dimension_reduction))
    

        if dim[0] % 2 == 1:
            dim = (dim[0] + 1, dim[1])
        
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)
        self.frame = frame
    

    def show(self):
        frame = self.frame
        
        dim = (int(frame.shape[1]*self.dimension_reduction),
               int(frame.shape[0]*self.dimension_reduction))
        cv2.imshow("frame", cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST))

        
        left_right_dim = (int(self.left.shape[1]*self.dimension_reduction),
                          int(self.left.shape[0]*self.dimension_reduction))

        cv2.imshow("left", cv2.resize(self.left, left_right_dim, interpolation = cv2.INTER_NEAREST))
        cv2.imshow("right", cv2.resize(self.right, left_right_dim, interpolation = cv2.INTER_NEAREST))
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            self.is_running = False

            
    def get_obs(self):
        self.get_frame()
        # Image in shape (H, W, 1) pixels value in [0, 255] because SB3 automatically scales the features for images
        observation = np.reshape(np.array(self.frame), 
                                 (self.frame.shape[0],self.frame.shape[1] , 1))
        
        asymmetry = self.get_asymmetry()
        return observation, asymmetry
    
    def get_asymmetry(self):
        
        H, W = self.frame.shape[0], self.frame.shape[1]

        horizon = round(H*30/56)
        bottom = round(H*0)

        self.left = self.frame[horizon:H-bottom, 1:W//2]
        self.right = self.frame[horizon:H-bottom, W:W//2:-1]

        self.left = cv2.blur(self.left, (2,2))
        self.right = cv2.blur(self.right, (2,2))

        flat_left = np.reshape(self.left, -1)/255
        flat_right = np.reshape(self.right, -1)/255

        total = np.concatenate((flat_left, flat_right))
        mean, std = total.mean(), total.std()

        std_offset = 0.07
        flat_left = (flat_left - mean)/(std + std_offset)
        flat_right = (flat_right - mean)/(std + std_offset)

        error = mean_absolute_error(flat_left, flat_right)

        gain = 8
        offset = -1.0

        x = gain*(error + offset)

        asymmetry = 1/(1 + np.exp(-x)) 

        return asymmetry

