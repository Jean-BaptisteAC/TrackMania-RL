from typing import TypeVar
from stable_baselines3.common.env_checker import check_env 
from env.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
T = TypeVar("T")

if __name__ == "__main__":
    env =  TrackmaniaEnv(action_space="controller")
    check_env(env)


    while True:
    
        mu, sigma = 0.2, 0.3 # mean and standard deviation
        gas = 1
        steer = np.random.normal(0, 0)
        action = np.clip([gas, steer], -1.0, 1.0)

        new_observation, reward, done, truncated, info = env.step(action)

        try:
            if keyboard.is_pressed("p"):
                print("Interrupt")
                break
        except:
            pass

    action = np.array([0, 0, 0, 0])
    env.step(action)
    env.close()