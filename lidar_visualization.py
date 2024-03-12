from src.environment.utils.GameCapture import Lidar_Vision

if __name__ == "__main__":
    
    i = 0
    lidar = Lidar_Vision()
    while lidar.is_running:
        lidar.get_frame()
        lidar.get_rays()
        obs, min_distance = lidar.get_obs()
        if i == 10:
            print("min distance:", round(min_distance, 2))

            distance_reward = abs(1 - min_distance/0.27)
            alpha = 1
            reward = - alpha * distance_reward ** 2

            print("reward:", round(reward, 2))
            i = 0  
        i += 1

        lidar.show()
