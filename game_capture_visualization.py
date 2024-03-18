from src.environment.utils.GameCapture import Image_Vision
import numpy as np

if __name__ == "__main__":
    
    flash = 0
    image_vision = Image_Vision()
    while image_vision.is_running:
        obs, asymmtry = image_vision.get_obs()
        image_vision.show()

        if flash == 3:
            print(asymmtry)
            flash = 0
        flash += 1 