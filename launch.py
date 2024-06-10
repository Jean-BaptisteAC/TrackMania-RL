from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
import time

if __name__ == "__main__":
    env =  TrackmaniaEnv(dimension_reduction=6, 
                         training_track="C01",
                         training_mode="time_optimization", 
                         is_testing = False, 
                         action_mode="human")

    i = 0

    while True:
    
        action = [0, 0, 0]

        new_observation, reward, done, truncated, info = env.step(action)

        i += 1
        if i == 5:
            if reward > 0:
                print("reward", reward)
            i = 0

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass

env.close()