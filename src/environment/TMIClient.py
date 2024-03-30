import time
import numpy as np

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
        self.last_action_step = 0

        self.time = None
        self.passed_checkpoint = False
        self.is_respawn = False
        self.is_finish = False

        self.restart_idle = True
        self.is_idle = False

        self.rewind_state = None
        self.start_state = None
        
    def on_registered(self, iface: TMInterface) -> None:
        iface.execute_command("press delete")
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface, _time: int):

        if _time == 0:
            self.start_state = iface.get_simulation_state()

        if self.is_respawn and self.rewind_state is not None and _time >= 0:
            iface.rewind_to_state(self.rewind_state)
            self.is_respawn = False

        self.sim_state = iface.get_simulation_state()

        current_action = {
            'sim_clear_buffer': True,  
            "steer":           int(np.clip(self.action[0]*65536, -65536, 65536)),
            "accelerate":      self.action[1] > 0.5, 
            "brake" :          self.action[2] > 0.5
            }
        
        if self.last_action_step >= 100:
            current_action = {
                'sim_clear_buffer': True,
                "steer":           0,
                "accelerate":      False, 
                "brake" :          False
                }
            self.restart_idle = True
            
            if self.is_idle is False:
                self.is_idle = True
                iface.execute_command("press delete")

        iface.set_input_state(**current_action)

        self.last_action_step += 1
        
    def on_checkpoint_count_changed(self, iface, current: int, target: int):

        if current >= 1:

            if current == target:
                self.passed_checkpoint = True
                iface.prevent_simulation_finish()
                self.time = self.sim_state.race_time/1000 # race time is in milliseconds
                self.is_finish = True

    def respawn(self, state):
        self.is_respawn = True 
        self.passed_checkpoint = False
        self.is_finish = False

        if state is None:
            self.rewind_state = self.start_state
        else:
            self.rewind_state = state

    def reset_last_action_timer(self):
        self.last_action_step = 0
        self.restart_idle = False
        self.is_idle = False

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

