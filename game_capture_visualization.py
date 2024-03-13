from src.environment.utils.GameCapture import Image_Vision
import numpy as np

if __name__ == "__main__":
    
    image_vision = Image_Vision()
    while image_vision.is_running:
        obs, _ = image_vision.get_obs()
        image_vision.show()

    print(obs.shape)