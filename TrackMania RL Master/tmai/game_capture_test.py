from env.utils.GameCapture import Lidar_Vision

if __name__ == "__main__":
    
    i = 0
    lidar = Lidar_Vision()
    while lidar.is_running:
        lidar.get_frame()
        lidar.get_rays()
        obs = lidar.get_obs()
        lidar.show()