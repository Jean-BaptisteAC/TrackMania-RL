from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
import time

if __name__ == "__main__":
    env =  TrackmaniaEnv(observation_space="image", 
                         dimension_reduction=6, 
                         training_track=None, 
                         training_mode="exploration", 
                         render_mode="human")

    obs, _ = env.reset()
    i = 0

    while True:
        
        action = [0, 0, 0]
        new_observation, reward, done, truncated, info = env.step(action)
        
        if i == 7:
            print(reward)
            i = 0
        i += 1

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass

env.close()

