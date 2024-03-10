from typing import TypeVar
from env.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
T = TypeVar("T")


if __name__ == "__main__":
    env =  TrackmaniaEnv(action_space="arrows")
    while True:
    
        n = np.random.randint(4)
        action = np.array([0, 0, 0, 0])
        action[n] = 1

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