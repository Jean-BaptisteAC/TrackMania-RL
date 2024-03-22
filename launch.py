from typing import TypeVar
from stable_baselines3.common.env_checker import check_env 
from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
T = TypeVar("T")

if __name__ == "__main__":
    env =  TrackmaniaEnv(observation_space="image", dimension_reduction=8)
    i = 0
    env.reset()
    while True:
        
        gas = np.random.normal() + 0.5
        steering = 10*np.random.normal()
        action = np.array([gas, steering])
        action = [0, 0]

        new_observation, reward, done, truncated, info = env.step(action)

        i += 1
        if i == 5:
            print(new_observation["physics"])
            i = 0

        if info["checkpoint_time"] is not False:
            print(info["checkpoint_time"])

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass


    action = np.array([0, 0])
    env.step(action)
    env.close()
