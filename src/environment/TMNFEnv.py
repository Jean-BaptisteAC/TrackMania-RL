import os
import pickle
from typing import TypeVar
import time
import numpy as np
import random
from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary, Dict

from .TMIClient import CustomClient
from .utils.GameCapture import Lidar_Vision, Image_Vision

from tminterface.interface import TMInterface

from scipy.interpolate import interp1d
from scipy.signal import argrelmin

import torch
 
ArrowsActionSpace = MultiBinary(4,)
ControllerActionSpace = Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
)
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class TrackmaniaEnv(Env):
    def __init__(
        self,
        observation_space: str = "image",
        dimension_reduction: int = 8
    ):
        self.action_space = ControllerActionSpace

        if observation_space == "lidar":
            self.observation_type = "lidar"
            self.viewer = Lidar_Vision()
            self.observation_space = Box(
                low=-1.0, high=1.0, shape=(17,), dtype=np.float64
            )
            

        elif observation_space == "image":
            self.observation_type = "image"
            self.viewer = Image_Vision(dimension_reduction=dimension_reduction)
            obs, _ = self.viewer.get_obs()
            image_shape = obs.shape
            self.observation_space = Dict(
                {"image": Box(low=0, high=255, shape=image_shape, dtype=np.uint8), 
                 "physics": Box(low=-1.0, high=1.0, shape=(6, ), dtype=np.float64)}
            )

        self.interface = TMInterface()
        self.client = CustomClient()
        self.interface.set_timeout(10_000)
        self.interface.register(self.client)

        while not self.interface.registered:
            time.sleep(0.1)

        self.max_race_time = 120_000
        self.first_init = True
        self.total_distance = 0
        self.last_time_step = 0

        self.train_steps = 3_000
        self.current_step = 0
        self.mode = "train"

        self.last_reset_time_step = 0

        self.init_centerline()

    def init_centerline(self):

        # init save_states for respawn
        run_folder = "track_data/Training_dataset_tech/run-2"
        state_files = list(filter(lambda x: x.startswith("state"), os.listdir(run_folder)))
        self.save_states = [pickle.load(open(os.path.join(run_folder, state_file), "rb")) for state_file in state_files]
        for state in self.save_states:
            state.dyna.current_state.linear_speed = np.array([0, 0, 0])

        # init centerline
        positions = pickle.load(open(os.path.join(run_folder, "positions.pkl"), "rb"))
        raw_points = [list(pos['position'].to_numpy()) for pos in positions]
        points = [raw_points[0]]
        for point in raw_points[1:]:
            if point != points[-1]:
                points.append(point)

        points = np.array(points)

        # Linear length along the line:
        distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )))
        distance = np.insert(distance, 0, 0)/distance[-1]

        interpolator =  interp1d(distance, points, kind='linear', axis=0)
        alpha = np.linspace(0, 1, 1000)
        self.centerline = interpolator(alpha)


    def reset(self, seed=0):

        if self.mode == "train":
            self.client.mode = "train"
            state = random.choice(self.save_states)
        else:
            self.client.mode = "eval"
            state = None
        
        self.client.respawn(state)
        
        screen_observation, _ = self.viewer.get_obs()
        observation = self.observation(screen_observation)
        info = {}

        self.total_distance = 0
        self.last_reset_time_step = self.state.race_time
        
        return observation, info

    def step(self, action):

        self.client.action = action
        self.client.reset_last_action_timer()
        
        screen_observation, distance_observation = self.viewer.get_obs()
        observation = self.observation(screen_observation)

        # Total distance in meters: (timestep in seconds) x (velocity in meters per second)
        self.total_distance += ((self.state.race_time - self.last_time_step) /1000) * (np.linalg.norm(self.velocity())/3.6)
        self.last_time_step = self.state.race_time

        # velocity_reward = np.linalg.norm(self.velocity())/100
        velocity_reward = np.log(1 + np.linalg.norm(self.velocity())/70)

        contact = self.state.scene_mobil.has_any_lateral_contact
        wall_penalty = float(contact)

        if self.observation_type == "lidar":
            distance_reward = abs(1 - distance_observation/0.27)
            alpha = 0.5
            reward = velocity_reward - (alpha * distance_reward ** 2) - wall_penalty

        elif self.observation_type == "image":
            distance_reward = distance_observation
            alpha = 0.5
            reward = velocity_reward - (alpha * distance_reward) - wall_penalty

        special_reward, done, info  = self.check_state()
        if special_reward is not None:
            reward = special_reward
        
        truncated = False

        update_done = self.update_env_mode(done)
        if update_done is not None:
            done = update_done

        # TEMPORARY: TESTING MINIMAL DISTANCE TO CENTERLINE
        reward = self.compute_centerline_distance()

        return observation, reward, done, truncated, info
    
    def close(self):
        self.interface.close()

    def update_env_mode(self, done):
        if self.mode == "train":
            self.current_step += 1
        
            if self.current_step == self.train_steps:
                self.mode = "eval"
                print("current mode:", self.mode)
                self.current_step = 0
                self.reset()
                return True
  
        elif self.mode == "eval" and done:
            self.mode = "train"
            print("current mode:", self.mode)
            self.reset()
            return True
        
        return None
        

    def check_state(self):
        special_reward = None 
        done = False
        info = {"checkpoint_time":False, 
                "total_distance":False}

        # Check for exit of the track
        if self.position[1] < 9.2:
            done = True
            special_reward = -20
            info["total_distance"] = self.total_distance
            self.reset()

        # Check for complete stop of the car
        if (self.race_time - self.last_reset_time_step) >= 1_000:
            if self.velocity()[2] < 1:
                done = True
                special_reward = -20
                info["total_distance"] = self.total_distance
                self.reset()

        # Check for finishing in the checkpoint
        if self.client.passed_checkpoint:
            special_reward = 100
            self.client.passed_checkpoint = False
            if self.client.is_finish:
                done = True    
                info["checkpoint_time"] = self.client.time
                info["total_distance"] = self.total_distance
                self.reset()

        # Check for contact with barriers in lidar mode
        if self.viewer.touch_boarder():
            done = True
            special_reward = -20
            info["total_distance"] = self.total_distance
            self.reset()
            
        # Time out when max episode duration is reached
        if self.race_time >= self.max_race_time :
            done = True
            special_reward = -20
            info["total_distance"] = self.total_distance
            self.reset()

        # Restart the simulation if client was idle due to SB3 update
        if self.client.restart_idle:
            done = True
            self.reset()

        if self.mode == "train":
            info["checkpoint_time"] = False
            info["total_distance"] = False
        
        return special_reward, done, info
    

    def observation(self, screen_observation):
        if self.observation_type == "lidar":
            physics = self.physics_instruments()
            observation = np.concatenate([screen_observation, physics])

        elif self.observation_type == "image":
            observation = {"image":screen_observation, 
                           "physics": torch.tensor(self.physics_instruments(), dtype=torch.float64)
            }
        return observation 
        
    
    def velocity(self):
        # Projection of the speed vector on the direction of the car
        # Track mania X, Y, Z is not the usual system: Y corresponds to the altitude.
        # Local velocity in m/s is: lateral, vertical, forward
        local_velocity = self.state.scene_mobil.current_local_speed
        local_velocity = np.array(list(local_velocity.to_numpy()))
        
        # Conversion from m/s to km/h
        local_velocity = local_velocity*3.6 
        return local_velocity
    
    def physics_instruments(self):
        forward_speed = self.velocity()[2]/1000
        lateral_speed = self.velocity()[0]/1000

        pitch_angle = self.state.yaw_pitch_roll[1]/np.pi
        roll_angle = self.state.yaw_pitch_roll[2]/np.pi

        turning_rate = self.state.scene_mobil.turning_rate

        ground_contact = self.has_ground_contact()

        return np.array([forward_speed, 
                         lateral_speed, 
                         pitch_angle, 
                         roll_angle,
                         turning_rate, 
                         ground_contact])

    def has_ground_contact(self):
        for wheel in self.state.simulation_wheels:
            if wheel.real_time_state.has_ground_contact:
                return 1.0
        return 0.0

    @property
    def state(self):
        return self.client.sim_state

    @property
    def position(self):
        return self.state.position

    @property
    def race_time(self):
        return self.state.race_time
    
    def distance_3D(self, x, y, z, x0, y0, z0):
        d_x = x - x0
        d_y = y - y0
        d_z = z - z0
        dis = np.sqrt( d_x**2 + d_y**2 + d_z**2)
        return dis
    
    def compute_centerline_distance(self):
        x = self.centerline[:,0]
        y = self.centerline[:,1]
        z = self.centerline[:,2]

        current_position = list(self.state.dyna.current_state.position.to_numpy())

        # compute distance
        dis = self.distance_3D(x, y, z, current_position[0], current_position[1], current_position[2])
        # find the minima
        min_idxs = argrelmin(dis)[0]
        # take the minimum
        glob_min_idx = min_idxs[np.argmin(dis[min_idxs])]

        # minimal distance to centerline
        min_d = dis[glob_min_idx]
        return min_d