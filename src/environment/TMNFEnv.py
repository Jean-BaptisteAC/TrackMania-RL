from typing import TypeVar
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary

from environment.TMIClient import ThreadedClient
from environment.utils.GameCapture import Lidar_Vision
from environment.utils.GameInteraction import ArrowInput, KeyboardInputManager, GamepadInputManager
from environment.utils.GameLaunch import GameLauncher
 
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
        action_space: str = "arrows",
        observation_size: int = 16,
    ):
        if action_space == "arrows":
            self.action_space = ArrowsActionSpace
            self.input_manager = KeyboardInputManager()
        elif action_space == "controller":
            self.action_space = ControllerActionSpace
            # self.input_manager = GamepadInputManager()

        self.special_input_manager = KeyboardInputManager()

        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(observation_size,), dtype=np.float64
        )
        game_launcher  = GameLauncher()
        if not game_launcher.game_started:
            game_launcher.start_game()
            print("game started")
            input("press enter when game is ready")
            time.sleep(4)
        
        self.viewer = Lidar_Vision()
        self.simthread = ThreadedClient()
        self.total_reward = 0.0
        self.current_race_time = 0
        self.max_race_time = 12_000
        self.first_init = True
    
        self.i_step = 0

    def close(self):
        self.simthread.iface.close()

    def step(self, action):

        self._continuous_action_to_command(action)
        
        self.viewer.get_frame()
        self.viewer.get_rays()
        self.viewer.show()
        
        velocity_reward, done = self.check_state()
        lidar, min_distance = self.viewer.get_obs()
        velocity = self.speedometer
        obs = np.concatenate([lidar, velocity])
        
        distance_reward = abs(1 - min_distance/0.27)
        alpha = 0.5

        reward = velocity_reward - alpha * distance_reward ** 2
        self.total_reward += reward

        truncated = False
        info = {}

        return obs, reward, done, truncated, info

    def check_state(self):
        done = False
        reward = self.reward

        if self.position[0] < 41.0:
            done = True
            reward = -20
            self.reset()

        elif self.viewer.touch_boarder():
            done = True
            reward = -20
            self.reset()
            
        elif (self.race_time - self.current_race_time) >= self.max_race_time :
            done = True
            self.reset()
        
        return reward, done
        
    def reset(self, seed=0):
        self.total_reward = 0.0
        if self.first_init:
            self.current_race_time = 0
            self.first_init = False
        else:
            self.current_race_time = self.race_time
        self._restart_race()
        
        self.viewer.get_frame()
        self.viewer.get_rays()
        lidar, _ = self.viewer.get_obs()
        velocity = self.speedometer
        obs = np.concatenate([lidar, velocity])
        info = {}
        
        return obs, info

    def action_to_command(self, action):
        if isinstance(self.action_space, MultiBinary):
            return self._discrete_action_to_command(action)
        elif isinstance(self.action_space, Box):
            return self._continuous_action_to_command(action)

    def _continuous_action_to_command(self, action):
        self.simthread.action = action
        
    def _discrete_action_to_command(self, action):
        commands = ArrowInput.from_discrete_agent_out(action)
        self.input_manager.play_inputs_no_release(commands)

    def _restart_race(self):
        self.special_input_manager.press_key(ArrowInput.RETURN)
        time.sleep(0)
        self.special_input_manager.release_key(ArrowInput.RETURN)

    @property
    def state(self):
        return self.simthread.data
    
    @property
    def speed(self):
        return self.state.display_speed

    @property
    def velocity(self):
        # Conversion from m/s to km/h
        velocity_norm = np.linalg.norm(self.state.velocity)*3.6 
        return velocity_norm
    
    @property
    def position(self):
        return self.state.position

    @property
    def race_time(self):
        return self.state.race_time

    @property
    def speedometer(self):
        return np.array([(self.velocity/1000 - 0.5)*2])

    @property
    def reward(self):
        reward = self.velocity/100
        return reward

