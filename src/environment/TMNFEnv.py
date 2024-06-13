import os
import pickle
from typing import TypeVar
import time
import numpy as np
import pandas as pd
import random
from gymnasium import Env
from gymnasium.spaces import Box, Dict

from .TMIClient import CustomClient
from .utils.GameCapture import Image_Vision

from tminterface.interface import TMInterface

from scipy.interpolate import interp1d

import torch
 
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

class TrackmaniaEnv(Env):
    """ Environment able to interact with Trackmania Nations Forever. 
    It purpose is to get game data, process them to feed the agent. It receives the agent's action and send it to the game.
    Meanwhile, it computes the reward.
    
    Args:
        observation_space (str): Type of observation space. Default is "image".
        dimension_reduction (int): Dimension reduction factor for the image observation space. Default is 6.
        training_track (str): Track used for training and data loading. Default is None.
        training_mode (str): Mode for exploration or time optimization reward shaping. Default is exploration.
            - exploration: log(velocity) reward with wall avoidance and center of track incentive
            - time_optimization: progression based reward only
        testing (bool): Bool for training or testing
        render_mode (str): Mode for rendering the game. Default is None.
            - None: No rendering
            - "human": Rendering in a window
        action_mode (str): Mode for action inputs. Default is None
            - None: agent inputs
            - "human": human inputs
    """
    def __init__(
        self,
        dimension_reduction: int = 6,
        training_track: str | None = None,
        training_mode: str = "exploration",
        is_testing: bool = False,
        render_mode: str | None = None,
        action_mode: str | None = None, 
    ):
        self.action_space = Box(
            low=np.array([-1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32
        )   
        
        self.observation_type = "image"
        self.viewer = Image_Vision(dimension_reduction=dimension_reduction)
        obs, _ = self.viewer.get_obs()
        image_shape = obs.shape
        self.observation_space = Dict(
            {"image": Box(low=0.0, high=255, shape=image_shape, dtype=np.uint8), 
                "physics": Box(low=-1.0, high=1.0, shape=(9, ), dtype=np.float64)}
        )

        self.interface = TMInterface()
        self.client = CustomClient(action_mode)
        self.interface.set_timeout(10_000)
        self.interface.register(self.client)

        while not self.interface.registered:
            time.sleep(0.1)

        self.training_track = training_track

        self.first_init = True
        self.total_distance = 0
        self.last_time_step = 0

        self.previous_centerline_time = [0]
        self.previous_centerline_positions = []

        self.last_reset_time_step = 0

        if self.training_track is not None:
            self.init_centerline()
            self.checkpoint_id = np.random.randint(len(self.save_states))
            self.episode_state = self.save_states[self.checkpoint_id]

        self.training_mode = training_mode
        if self.training_mode == "exploration":
            # For exploration only
            self.episode_length = 400
 
        elif self.training_mode == "time_optimization":
            # For time_optimization only
            self.episode_length = 2048

        self.episode_step = 0
        
        self.is_testing = is_testing
        self.render_mode = render_mode
        self.action_mode = action_mode

        self.collision_duration = 1
        self.collision_timer = self.collision_duration

    def init_centerline(self):

        # init save_states for respawn
        run_folder = "track_data/" + self.training_track + "/run-1"
        state_files = list(filter(lambda x: x.startswith("state"), os.listdir(run_folder)))
        self.save_states = [pickle.load(open(os.path.join(run_folder, state_file), "rb")) for state_file in state_files]

        # init centerline
        positions = pickle.load(open(os.path.join(run_folder, "positions.pkl"), "rb"))

        raw_points = [list(pos['position'].to_numpy()) for pos in positions]

        # smoothing
        df = pd.DataFrame(raw_points)
        ema = df.ewm(com=40).mean()
        raw_points = ema.values.tolist()
            
        # remove duplicates:
        points = [raw_points[0]]
        for point in raw_points[1:]:
            if point != points[-1]:
                points.append(point)
            else:
                for i in range(len(point)):
                    point[i] += 0.01
                points.append(point)
        points = np.array(points)

        # Time along the track:
        time = np.linspace(0, 1, len(points))

        self.interpolator =  interp1d(time, points, kind='slinear', axis=0)

        self.alpha = np.linspace(0, 1, len(points))
        self.centerline = self.interpolator(self.alpha)
        self.finish_time = positions[-1]["time"]/1000

        # Progression percentage along the track:
        distance = np.cumsum(np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1)))
        distance = np.insert(distance, 0, 0)/distance[-1]
        self.percentage_interpolator = interp1d(distance, points, kind='slinear', axis=0)
        self.percentage_progression_centerline = self.percentage_interpolator(self.alpha)

    def reset(self, seed=0):

        if self.training_track is not None:
            state = self.episode_state
        else:
            state = None
        self.client.respawn(state)

        if self.training_mode == "time_optimization":
            self.episode_step = 0
        
        screen_observation, _ = self.viewer.get_obs()
        observation = self.observation(screen_observation)
        info = {}

        self.last_reset_time_step = 0
        self.last_time_step = self.state.race_time

        self.previous_centerline_time = [0]
        self.previous_centerline_positions = []
        
        return observation, info
    
    def render(self):
        if self.render_mode == "human":
            self.viewer.show()

    def step(self, action):

        self.client.action = action
        self.last_reset_time_step += 1 

        # Total distance in meters: (timestep in seconds) x (velocity in meters per second)
        self.total_distance += ((self.state.race_time - self.last_time_step) /1000) * (np.linalg.norm(self.velocity())/3.6)
        self.last_time_step = self.state.race_time

        if self.training_track is not None:
            min_d, eq_time = self.compute_centerline_distance()
            self.min_d = min_d
            self.eq_time = eq_time

        # Add sleep in env for 20 FPS target
        time.sleep(0.0165)
        self.episode_step += 1
        
        screen_observation, distance_observation = self.viewer.get_obs()
        self.render()
        observation = self.observation(screen_observation)

        if self.training_mode == "exploration":
            reward = self.exploration_reward(distance_observation)

        elif self.training_mode == "time_optimization":
            reward = self.time_optimization_reward()

        special_reward, done, truncated, info  = self.check_state()
        if special_reward is not None:
            reward = special_reward

        self.client.reset_last_action_timer()

        reward = np.clip(reward, -10, 10)

        return observation, reward, done, truncated, info
        
    
    def close(self):
        self.interface.close()

    def exploration_reward(self, distance_observation):

        velocity_reward = np.log(1 + max(0, self.velocity()[2])/70)

        if self.client.collision:
            self.collision_timer = self.collision_duration
        
        wall_penalty = int(self.collision_timer >= 0)

        self.collision_timer -= 1
        self.client.reset_collision()
                           

        if self.observation_type == "lidar":
            distance_reward = abs(1 - distance_observation/0.27)
            alpha = 0.5
            reward = velocity_reward - (alpha * distance_reward ** 2) - wall_penalty

        elif self.observation_type == "image":
  
            distance_reward = distance_observation
            alpha = 1.0
            reward = velocity_reward - (alpha * distance_reward) - wall_penalty
        
        return reward

    
    def time_optimization_reward(self):

        if self.client.collision:
            self.collision_timer = self.collision_duration
        
        wall_penalty = int(self.collision_timer >= 0)

        self.collision_timer -= 1
        self.client.reset_collision()
        
        progress_reward = self.eq_time - self.previous_centerline_time[0]
        self.previous_centerline_time.append(self.eq_time)
        if len(self.previous_centerline_time) > 3:
            self.previous_centerline_time.pop(0)

        # distance_reward = self.min_d/11.6 # Average Radius of the road

        collision_reward = min(1, wall_penalty*((np.linalg.norm(self.velocity())/300)**2))
        reward = 4*progress_reward - 2*collision_reward
        return reward


    def check_state(self):
        special_reward = None 
        done = False
        truncated = False
        info = {"checkpoint_time":False,
                "total_distance":False, 
                "percentage_progress":False}

        # Check for exit of the track
        if self.position[1] < 9.2:
            done = True
            special_reward = -10

            if self.training_track is None or self.is_testing:
                info["total_distance"] = self.total_distance
                info["percentage_progress"] = self.compute_centerline_percentage_progression()
                self.total_distance = 0

            self.reset()

        # Check for distance from centerline 
        if self.training_track is not None and self.is_testing is False:
            if self.min_d > 23:
                done = True
                special_reward = -10

                if self.training_track is None or self.training_mode == "time_optimization":
                    info["total_distance"] = self.total_distance
                    info["percentage_progress"] = self.compute_centerline_percentage_progression()
                    self.total_distance = 0

                self.reset()

        # Check for complete stop of the car
        if self.last_reset_time_step >= 20:
            if self.velocity()[2] < 1:
                done = True
                special_reward = -10

                if self.training_track is None or self.training_mode == "time_optimization":
                    info["total_distance"] = self.total_distance
                    info["percentage_progress"] = self.compute_centerline_percentage_progression()
                    self.total_distance = 0

                self.reset()

        # Check for finishing in the checkpoint
        if self.client.passed_checkpoint:
            self.client.passed_checkpoint = False
            if self.client.is_finish:
                done = True    
                info["checkpoint_time"] = self.client.time

                if self.training_track is None or self.training_mode == "time_optimization":
                    special_reward = 10
                    info["total_distance"] = self.total_distance
                    info["percentage_progress"] = self.compute_centerline_percentage_progression()
                    self.total_distance = 0

                self.reset()

        # Restart the simulation if client was idle due to SB3 update
        if self.client.restart_idle:
            done = True
            truncated = True
            self.reset()

        # Time out when max episode duration is reached 
        if self.training_track is not None or self.training_mode == "time_optimization":
            if self.is_testing is False:
                if self.episode_step >= self.episode_length:
                    done = True
                    self.episode_step = 0
                    self.episode_state = random.choice(self.save_states)
                    info["total_distance"] = self.total_distance
                    info["percentage_progress"] = self.compute_centerline_percentage_progression()
                    self.total_distance = 0
                    truncated = True
                    self.reset()
        
        # Check for reverse traveling during time_optimization
        if self.last_reset_time_step >= 20:
            if self.training_mode == "time_optimization":
                progress_reward = self.eq_time - self.previous_centerline_time[0]
                if progress_reward < 0:
                    done = True
                    special_reward = -10


                    if self.training_track is None or self.training_mode == "time_optimization":
                        info["total_distance"] = self.total_distance
                        info["percentage_progress"] = self.compute_centerline_percentage_progression()
                        self.total_distance = 0

                    self.reset()

        return special_reward, done, truncated, info


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

        is_sliding = self.state.scene_mobil.is_sliding

        input_brake = int(self.state.scene_mobil.input_brake > 0.5)

        ground_contact = self.has_ground_contact()

        lateral_contact = float(self.state.scene_mobil.has_any_lateral_contact)

        return np.array([forward_speed,
                         lateral_speed,
                         pitch_angle,
                         roll_angle,
                         turning_rate,
                         is_sliding, 
                         input_brake,
                         ground_contact,
                         lateral_contact])

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

        self.previous_centerline_positions.append(self.position)
        if len(self.previous_centerline_positions) >= 10:
            self.previous_centerline_positions.pop(0)
        current_position = np.array(self.previous_centerline_positions).mean(axis=0)

        # compute distance
        dis = self.distance_3D(x, y, z, current_position[0], current_position[1], current_position[2])
        # find the minima
        glob_min_idx = np.argmin(dis)

        # minimal distance to centerline
        min_d = dis[glob_min_idx]

        # equivalent time of the centerline
        eq_time = self.alpha[glob_min_idx]*self.finish_time

        # DOUBLE PRECISION
        min_index = max(0, glob_min_idx - 1)
        max_index = min(len(self.alpha) -1, glob_min_idx + 1)
        lower = self.alpha[min_index]
        higher = self.alpha[max_index]

        alpha2 = np.linspace(lower, higher, len(self.alpha))
        coords2 = self.interpolator(alpha2)
        x = coords2[:,0]
        y = coords2[:,1]
        z = coords2[:,2]

        dis = self.distance_3D(x, y, z, current_position[0], current_position[1], current_position[2])

        # find the minima
        glob_min_idx = np.argmin(dis)
        # distance
        min_d = dis[glob_min_idx]

        # Asymmetric distance 
        d_x = x[glob_min_idx] - current_position[0]
        d_y = (y[glob_min_idx] - current_position[1])/2
        d_z = z[glob_min_idx] - current_position[2]
        min_d = np.linalg.norm([d_x, d_y, d_z])

        eq_time = alpha2[glob_min_idx]*self.finish_time

        return min_d, eq_time

    def compute_centerline_percentage_progression(self):
        x = self.percentage_progression_centerline[:,0]
        y = self.percentage_progression_centerline[:,1]
        z = self.percentage_progression_centerline[:,2]

        # compute distance
        dis = self.distance_3D(x, y, z, self.position[0], self.position[1], self.position[2])
        # find the minima
        glob_min_idx = np.argmin(dis)

        associated_progress = glob_min_idx/len(self.alpha)

        return associated_progress