import time
import numpy as np
from threading import Lock, Thread

from tminterface.client import Client
from tminterface.interface import TMInterface

import keyboard

class CustomClient(Client):
    """
    Client for a TMInterface instance.
    """

    def __init__(self):
        super().__init__()
        self.sim_state = None
        self.action = [0, 0, 0]
        self.last_action_timer = 0

        self.time = None

        self.is_init = False
        self.init_state = None
        self.passed_checkpoint = False
        self.is_respawn = True
        self.is_finish = False

        self.restart_idle = True

        self.mode = "train"
        self.train_state = None
        
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface, _time: int):

        if _time == 0:
            self.init_state = iface.get_simulation_state()

        if not self.is_init:
            iface.respawn()
            self.is_init = True
            self.time = None

        elif self.is_respawn and self.init_state:

            if self.mode == "eval":
                iface.rewind_to_state(self.init_state)
            elif self.mode == "train":
                iface.rewind_to_state(self.train_state)
            
            self.is_respawn = False

        self.sim_state = iface.get_simulation_state()

        current_action = {
            'sim_clear_buffer': True,  
            "steer":           int(np.clip(self.action[0]*65536, -65536, 65536)),
            "accelerate":      self.action[1] > 0.5, 
            "brake" :          self.action[2] > 0.5
            }
        
        if self.sim_state.race_time - self.last_action_timer > 1_000:
            current_action = {
                'sim_clear_buffer': True,
                "steer":           0,
                "accelerate":      False, 
                "brake" :          False
                }
            self.restart_idle = True


        iface.set_input_state(**current_action)
       
        
    def on_checkpoint_count_changed(self, iface, current: int, target: int):

        if current >= 1:
            # self.passed_checkpoint = True

            if current == target:
                iface.prevent_simulation_finish()
                self.time = self.sim_state.race_time/1000 # race time is in milliseconds
                self.is_finish = True

    def respawn(self, state):
        self.is_respawn = True 
        self.passed_checkpoint = False
        self.is_finish = False
        self.train_state = state

    def reset_last_action_timer(self):
        self.last_action_timer = self.sim_state.race_time
        self.restart_idle = False

if __name__ == "__main__":
    
    
    interface = TMInterface()
    client = CustomClient()
    interface.register(client)
    while interface.running:
        time.sleep(0)
        client.action[1] = np.random.normal()
        try:
            if keyboard.is_pressed("q"):
                print("Interrupt")
                break
        except:
            pass
    interface.close()

