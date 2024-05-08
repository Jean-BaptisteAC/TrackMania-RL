import os
import time
import pickle
import keyboard
import numpy as np
import random

import tminterface as tmi
from tminterface.interface import TMInterface, Client

# USEFUL FUNCTIONS

def discrete_to_continuous(n):

    """
    Equivalents:
    
    0: no action
    1: left
    2: left + acceleration
    3: acceleration
    4: right + acceleration
    5: right
    """

    current_action = {
    'sim_clear_buffer': True,  
    "steer":           0,
    "accelerate":      False, 
    "brake" :          False
    }
    
    if n == 1:
        current_action["steer"] = -65536
    if n == 2:
        current_action["steer"] = -65536
        current_action["accelerate"] = True
    if n == 3:
        current_action["accelerate"] = True
    if n == 4:
        current_action["steer"] = 65536
        current_action["accelerate"] = True
    if n == 5:
        current_action["steer"] = 65536
        
    return current_action


# ABSTRACT CLIENT

class AbstractClient(Client):

    def __init__(self):
        super().__init__()
        self.dna = None
        self.final_state = None
        self.start_state = None
        self.is_finish = False
        self.finish_dna = None
        
    def on_registered(self, iface: TMInterface) -> None:
        iface.execute_command("press delete")
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface, _time: int):
        self.action(iface, _time)

    def on_checkpoint_count_changed(self, iface, current: int, target: int):
        if current >= 1 and current == target:
            self.finish(iface)
            self.is_finish = True
            self.finish_dna = self.dna.copy()

    def reset_detection(self, _time, state):
        
        if state.position[1] < 9.2:
            return True
    
        if _time >= 500:
            local_velocity = state.scene_mobil.current_local_speed
            local_velocity = np.array(list(local_velocity.to_numpy()))
            local_velocity = local_velocity*3.6 
            if local_velocity[2] < 1:
                return True

        if state.scene_mobil.has_any_lateral_contact:
            return True
    
        return False

    def action(self, iface, _time: int):
        if _time >= 0 and _time < len(self.dna)*1000:
            if _time == 0:
                self.start_state = iface.get_simulation_state()
            
            action = self.dna[int(np.floor(_time/1_000))]
            command = discrete_to_continuous(action)
            iface.set_input_state(**command)
            
            if self.reset_detection(_time, iface.get_simulation_state()):
                self.finish(iface)
                
        if _time == len(self.dna)*1000:
            self.finish(iface)

    def finish(self, iface):
        self.final_state = iface.get_simulation_state()
        iface.rewind_to_state(self.start_state)

class SimpleClient(Client):

    def __init__(self, dna):
        super().__init__()
        self.dna = dna
        
    def on_registered(self, iface: TMInterface) -> None:
        iface.execute_command("press delete")
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface, _time: int):
        self.action(iface, _time)


    def action(self, iface, _time: int):
        if _time >= 0 and _time < len(self.dna)*1000:
            if _time == 0:
                self.start_state = iface.get_simulation_state()
            
            action = self.dna[int(np.floor(_time/1_000))]
            command = discrete_to_continuous(action)
            iface.set_input_state(**command)
                
        if _time == len(self.dna)*1000:
            self.finish(iface)

    def finish(self, iface):
        iface.rewind_to_state(self.start_state)


# TRAINING CLIENT AND REPLAY CLIENT

class TrainingClient(AbstractClient):

    def __init__(self, horizon=5, n_trials=100):
        super().__init__()
        self.generate_dna()
        self.horizon = horizon
        self.max_trials = n_trials
        self.n_trial = 1
        self.memory = np.array([])

        self.best_gene = self.dna
        self.best_perf = 0
        self.mid_state = None

    def generate_dna(self):
        self.dna = np.random.randint(low=2, high=5, size=10)

    def on_run_step(self, iface, _time: int):
        self.action(iface, _time)
        if _time == (len(self.memory) + self.horizon)*1000:
            self.mid_state = iface.get_simulation_state()
            
    def finish(self, iface):
        self.n_trial += 1
        self.final_state = iface.get_simulation_state()

        # OBECTIVE FUNCTION CONDITION
        if self.final_state.position[0] > self.best_perf:
            self.best_gene = self.dna
            self.best_perf = self.final_state.position[0]

        if self.n_trial == self.max_trials:
            if self.mid_state is not None:
                self.start_state = self.mid_state
                self.memory = np.concatenate([self.memory, self.best_gene[:self.horizon]])
            self.n_trial = 0
            print(self.memory)
        
        iface.rewind_to_state(self.start_state)            
        self.generate_dna()
        
class ReplayClient(AbstractClient):

    def __init__(self, dna): 
        super().__init__()
        self.dna = dna
        
    def finish(self, iface):
        self.final_state = iface.get_simulation_state()
        print(self.final_state.position)
        iface.rewind_to_state(self.start_state)


if __name__ == "__main__":

    interface = TMInterface()
    # client = TrainingClient(horizon=2, n_trials=100)
    client = SimpleClient(dna=np.array([3,3,3]))

    interface.register(client)
    print("Start")

    while True:
        time.sleep(0.001)

        if keyboard.is_pressed("q"):
            print("Keybord Interrupt")
            break

    # if client.is_finish:
    #     best_memory = client.memory + client.finish_dna
    # else:
    #     best_memory = client.memory

    interface.close()
    # best_dna = client.memory
    # print(best_dna)