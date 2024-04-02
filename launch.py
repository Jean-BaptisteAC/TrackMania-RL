from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard
import time

if __name__ == "__main__":
    env =  TrackmaniaEnv(observation_space="image", 
                         dimension_reduction=6, 
                         training_track=None, 
                         render_mode= "human")
    env.reset()
    i = 0
    while True:
        
        action = [0, 0, 0]
        new_observation, reward, done, truncated, info = env.step(action)

        if done:
            info
        
        if i == 5:
            print(reward)
            i = 0
        i += 1

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass

    action = np.array([0, 0, 0])
    env.step(action)
    env.close()