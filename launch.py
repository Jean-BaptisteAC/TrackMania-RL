from typing import TypeVar
from stable_baselines3.common.env_checker import check_env 
from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
T = TypeVar("T")

if __name__ == "__main__":
    env =  TrackmaniaEnv(action_space="controller", observation_space="image")

    env.reset()
    while True:
        
        gas = 0.5
        steering = np.clip(np.random.normal(), -1, 1)
        # action = np.array([gas, steering])

        action = np.array([1, 0])

        new_observation, reward, done, truncated, info = env.step(action)   

        try:
            if keyboard.is_pressed("p"):
                print("Interrupt")
                break
        except:
            pass

    action = np.array([0, 0])
    env.step(action)
    env.close()