from src.environment.TMNFEnv import TrackmaniaEnv
import numpy as np
import keyboard

if __name__ == "__main__":
    env =  TrackmaniaEnv(observation_space="image", dimension_reduction=8)
    env.reset()
    while True:
        
        action = [0, 0]
        new_observation, reward, done, truncated, info = env.step(action)

        # if info["total_distance"] is not False:
        #     print(info["total_distance"])

        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass


    action = np.array([0, 0])
    env.step(action)
    env.close()
