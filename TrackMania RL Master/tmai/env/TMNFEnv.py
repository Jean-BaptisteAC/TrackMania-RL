from typing import TypeVar
import time
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, MultiBinary

from env.TMIClient import ThreadedClient
from env.utils.GameCapture import Lidar_Vision
from env.utils.GameInteraction import ArrowInput, KeyboardInputManager
from env.utils.GameLaunch import GameLauncher


ArrowsActionSpace = MultiBinary(4,)  # none up down right left
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
        n_rays: int = 16,
    ):

        self.action_space = (ArrowsActionSpace)
        self.action_size = 7
        self.observation_space = Box(
            low=-1.0, high=1.0, shape=(n_rays,), dtype=np.float64
        )
        self.input_manager = (KeyboardInputManager())

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
        self.command_frequency = 50
        self.previous_step_time = time.time()
        self.step_time_memory = []
        self.first_init = True

    def close(self):
        self.simthread.iface.close()

    def step(self, action):

        self.action_to_command(action)
        
        self.viewer.get_frame()
        self.viewer.get_rays()
        self.viewer.show()
        
        velocity_reward, done = self.check_state()
        lidar = self.viewer.get_obs()
        velocity = self.speedometer
        obs = np.concatenate([lidar, velocity])
        
        reward = velocity_reward 
        self.total_reward += reward

        truncated = False
        info = {}

        return obs, reward, done, truncated, info

    def check_state(self):
        done = False
        reward = self.reward

        if self.position[0] < 41.0:
            done = True
            reward = -100
            self.reset()

        elif self.viewer.touch_boarder():
            done = True
            reward = -100
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
        lidar = self.viewer.get_obs()
        velocity = self.speedometer
        obs = np.concatenate([lidar, velocity])
        info = {}
        
        return obs, info

    def action_to_command(self, action):
        if isinstance(self.action_space, MultiBinary):
            return self._discrete_action_to_command(action)
        
    def _discrete_action_to_command(self, action):
        commands = ArrowInput.from_discrete_agent_out(action)
        self.input_manager.play_inputs_no_release(commands)

    def _restart_race(self):
        self.input_manager.press_key(ArrowInput.RETURN)
        time.sleep(0)
        self.input_manager.release_key(ArrowInput.RETURN)

    @property
    def state(self):
        return self.simthread.data
    
    @property
    def speed(self):
        return self.state.display_speed

    @property
    def velocity(self):
        return self.state.velocity
    
    @property
    def position(self):
        return self.state.position
    
    @property
    def simulation_wheels(self):
        return self.state.simulation_wheels

    @property
    def race_time(self):
        return self.state.race_time

    @property
    def speedometer(self):
        return np.array([(self.speed/1000 - 0.5)*2])

    @property
    def reward(self):
        reward = self.speed/100
        return reward