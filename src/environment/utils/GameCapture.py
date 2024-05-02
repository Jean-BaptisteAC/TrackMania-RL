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

class Lidar_Vision():
    def __init__(self):
        self.hwnd = wind32.FindWindow(None, os.getenv('GAME_WINDOW_NAME'))

        left, top, right, bottom = wind32.GetWindowRect(self.hwnd)
        bounding_box = left + 10, top + 40, right -10, bottom-10
        with mss() as sct:
            frame = np.array(sct.grab(bounding_box))

        dim = (int(frame.shape[1]/2),int(frame.shape[0]/2)) 
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
        self.angles = [0, 15, 30, 45, 60, 75, 85, 90, \
                         -15, -30, -45, -60, -75, -85, -90]
            
        self.get_frame()
        self.rays = None
        self.ref_point_front = (len(self.frame[0]) // 2, len(self.frame)-105)
        self.horizon_max = len(self.frame)/2 -30
        self.distance_max = 170
        self.is_running = True

    def get_frame(self):
        
        left, top, right, bottom = wind32.GetWindowRect(self.hwnd)
        bounding_box = left + 10, top + 40, right -10, bottom-10
        with mss() as sct:
            frame = np.array(sct.grab(bounding_box))
        dim = (int(frame.shape[1]/2),int(frame.shape[0]/2)) 
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        # _, frame = cv2.threshold(frame[:, :, 0], 110, 255, cv2.THRESH_BINARY)
        frame = cv2.GaussianBlur(frame, (7, 7), 0)
        frame = cv2.threshold(frame[:, :, 0], 130, 255, cv2.THRESH_BINARY)[1]
        
        self.frame = frame

    def get_rays(self):
    
        frame = self.frame
        rays = []
        for angle in self.angles:
            direction = angle*2*np.pi/360
            dx = np.sin(direction)
            dy = np.cos(direction)
            cur_x = self.ref_point_front[0]
            cur_y = self.ref_point_front[1]
    
            n = 0
            while n<self.distance_max:
                cur_x += dx
                cur_y -= dy
                n += 1
                if not self.is_inbouds(int(cur_x), int(cur_y), frame) \
                    or frame[int(cur_y)][int(cur_x)] == 0:
                    break

                if (cur_y) < self.horizon_max:
                    break

            rays.append((int(cur_x), int(cur_y)))

        self.rays = rays
    
    def is_inbouds(self, x, y, frame):
        return x >= 0 and x < len(frame[0]) and y >= 0 and y < len(frame)

    def touch_boarder(self):
        for ray in self.rays:
            n = distance.euclidean(self.ref_point_front, ray)
            if n <= 1:
                return True
        return False
    
    def show(self):
        frame = self.frame
        rays = self.rays
        
        for ray in rays:
            cv2.line(frame, self.ref_point_front, ray, (0, 0, 0), 1)
        
        dim = (int(frame.shape[1]*1.5),int(frame.shape[0]*1.5)) 
        cv2.imshow("frame", cv2.resize(frame, dim))
        
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            cv2.destroyAllWindows()
            self.is_running = False
            
    def get_obs(self):
        self.get_frame()
        self.get_rays()
        distances = []
        for ray in self.rays:
            distances.append(distance.euclidean(
                self.ref_point_front, ray)/self.distance_max)
        distances = np.array(distances)
        min_distance = min(distances)
        distances = (distances - 0.5)*2
        return np.array(distances), min_distance
    

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

    # No restart when touching the boarders
    def touch_boarder(self):
        return False
    

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

