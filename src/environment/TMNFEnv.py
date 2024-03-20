from typing import TypeVar
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary, Dict

from .TMIClient import ThreadedClient
from .utils.GameCapture import Lidar_Vision, Image_Vision
from .utils.GameLaunch import GameLauncher

import torch
 
ArrowsActionSpace = MultiBinary(4,)
ControllerActionSpace = Box(
    low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), shape=(2,), dtype=np.float32
)
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


class TrackmaniaEnv(Env):
    """
    Gym env interfacing the game.
    Observations are the rays of the game viewer.
    Controls are the arrow keys or the gas and steer.
    """
    def __init__(
        self,
        action_space: str = "controller",
        observation_space: str = "lidar",
    ):
        if action_space == "arrows":
            self.action_space = ArrowsActionSpace
        elif action_space == "controller":
            self.action_space = ControllerActionSpace

        if observation_space == "lidar":
            self.observation_type = "lidar"
            self.observation_space = Box(
                low=-1.0, high=1.0, shape=(16,), dtype=np.float64
            )
            self.viewer = Lidar_Vision()

        elif observation_space == "image":
            self.observation_type = "image"

            self.observation_space = Dict(
                {"image": Box(low=0, high=255, shape=(56, 158, 1), dtype=np.uint8), 
                 "physics": Box(low=-1.0, high=1.0, shape=(1, ), dtype=np.float64)}
            )
                
            self.viewer = Image_Vision()

        game_launcher  = GameLauncher()
        if not game_launcher.game_started:
            game_launcher.start_game()
            print("game started")
            input("press enter when game is ready")
            time.sleep(4)
        
        self.simthread = ThreadedClient()
        self.current_race_time = 0
        self.max_race_time = 25_000
        self.first_init = True
        self.i_step = 0

    def close(self):
        self.simthread.iface.close()

    def step(self, action):

        self._continuous_action_to_command(action)
        
        
        velocity_reward, done, info  = self.check_state()
        screen_observation, distance_observation = self.viewer.get_obs()
        self.viewer.show()

        observation = self.observation(screen_observation)

        contact = self.state.scene_mobil.has_any_lateral_contact
        if contact:
            wall_penalty = 1.0
        else: 
            wall_penalty = 0.0

        if self.observation_type == "lidar":
            distance_reward = abs(1 - distance_observation/0.27)
            alpha = 0.5
            reward = velocity_reward - (alpha * distance_reward ** 2) - wall_penalty

        elif self.observation_type == "image":
            distance_reward = distance_observation
            alpha = 0.5
            reward = velocity_reward - (alpha * distance_reward) - wall_penalty
        
        truncated = False

        return observation, reward, done, truncated, info

    def check_state(self):
        done = False

        info = {"checkpoint_time":False}
        reward = self.velocity()/100

        # Check for exit of the track
        if self.position[1] < 9.2:
            done = True
            reward = -20
            self.reset()

        # Check for complete stop of the car
        elif (self.race_time - self.current_race_time) >= 1_000:
            if self.velocity() < 1:
                done = True
                reward = -20
                self.reset()

        # Check for finishing in the checkpoint
        elif self.simthread.client.passed_checkpoint:
            done = True
            reward = 0
            info = {"checkpoint_time":self.simthread.client.time}
            self.reset()
            self.simthread.client.passed_checkpoint = False
            

        # Check for contact with barriers in lidar mode
        elif self.viewer.touch_boarder():
            done = True
            reward = -20
            self.reset()
            
        # Time out when max episode duration is reached
        elif (self.race_time - self.current_race_time) >= self.max_race_time :
            done = True
            self.reset()
        
        return reward, done, info
    

    def observation(self, screen_observation):
        if self.observation_type == "lidar":
            velocity = self.speedometer()
            observation = np.concatenate([screen_observation, velocity])

        elif self.observation_type == "image":
            observation = {"image":screen_observation, 
                           "physics": torch.tensor(np.array([self.speedometer()]), dtype=torch.float64)
            }
        return observation 
        
    def reset(self, seed=0):
        if self.first_init:
            self.current_race_time = 0
            self.first_init = False
        else:
            self.current_race_time = self.race_time
        self._restart_race_command()
        
        screen_observation, _ = self.viewer.get_obs()
        observation = self.observation(screen_observation)
        info = {}
        
        return observation, info

    def action_to_command(self, action):
        if isinstance(self.action_space, MultiBinary):
            return self._discrete_action_to_command(action)
        elif isinstance(self.action_space, Box):
            return self._continuous_action_to_command(action)

    def _continuous_action_to_command(self, action):
        self.simthread.action = action

    def _restart_race_command(self):
        self.simthread.client.respawn()

    # Computation of the direction vector of the car
    def vehicle_normal_vector(self):
        yaw_pitch_roll = self.state.yaw_pitch_roll

        # Track mania X, Y, Z is not the usual system: Y corresponds to the altitude.
        # We do not take into account the roll as it has no impact on the direction of the car
        orientation = [np.sin(yaw_pitch_roll[0]),
                    -np.sin(yaw_pitch_roll[1]), 
                    np.cos(yaw_pitch_roll[0])]
        orientation = orientation / np.linalg.norm(orientation)

        return orientation
    
    def velocity(self):
        
        # Projection of the speed vector on the direction vector of the car
        velocity_oriented = np.dot(self.vehicle_normal_vector(), 
                                 self.state.velocity)
        
        # Conversion from m/s to km/h
        velocity_oriented = velocity_oriented*3.6 
        return velocity_oriented
    
    def speedometer(self):
        return np.array([(self.velocity()/1000 - 0.5)*2])

    @property
    def state(self):
        return self.simthread.data

    @property
    def position(self):
        return self.state.position

    @property
    def race_time(self):
        return self.state.race_time