from src.environment.utils.GameCapture import Image_Vision
import numpy as np

if __name__ == "__main__":
    
    image_vision = Image_Vision(dimension_reduction=8)
    while image_vision.is_running:
        obs, asymmetry = image_vision.get_obs()
        image_vision.show()