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
        
        gas = np.random.normal() + 0.5
        steering = np.random.normal()
        action = np.array([gas, steering])
        action = [1, 0.1*steering]

        new_observation, reward, done, truncated, info = env.step(action)   

        if info["checkpoint_time"] is not False:
            print(info["checkpoint_time"])

        try:
            if keyboard.is_pressed("p"):
                print("Interrupt")
                break
        except:
            pass

    print(env.simthread.client.init_state)

    action = np.array([0, 0])
    env.step(action)
    env.close()