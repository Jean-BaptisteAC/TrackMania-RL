from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
import time

if __name__ == "__main__":
    env =  TrackmaniaEnv(observation_space="image", 
                         dimension_reduction=6, 
                         training_track=None, 
                         training_mode="exploration", 
                         render_mode=None, 
                         action_mode="human")

    while True:
        
        steer = np.random.random()*2 - 1 
        acceleration = np.random.random()
        action = [steer, acceleration, 0]

        new_observation, reward, done, truncated, info = env.step(action)

        if done: 
            print(info)

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass

env.close()