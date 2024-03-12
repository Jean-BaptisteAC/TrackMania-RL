import ctypes
import subprocess
from threading import Thread
import time

import os
from dotenv import load_dotenv
load_dotenv()

class GameLauncher:
    def __init__(
        self,
        game_path = os.getenv('GAME_PATH'),
        game_dir = os.getenv('GAME_DIR'),
        game_window_name = os.getenv('GAME_WINDOW_NAME') 
        ) -> None:
        
        print(game_path, game_dir, game_window_name)

        self.game_path = game_path
        self.game_dir = game_dir
        self.game_window_name = game_window_name
        self.game_thread = Thread(target=self.game_thread, daemon=False)
        
    def game_thread(self):
        subprocess.Popen(self.game_path, cwd=self.game_dir)

        while True:
            time.sleep(0)
            
    @property
    def game_started(self)->bool:
        try:
            hwnd = ctypes.windll.user32.FindWindowW(None, self.game_window_name)
            if hwnd == 0: raise Exception("game not started")
        except:
            return False
        return True
            
    def start_game(self):
        if not self.game_started:
            self.game_thread.start()
            
        else:
            print("game already started")
            
if __name__ == "__main__":
    game_launcher = GameLauncher()
    game_launcher.start_game()
