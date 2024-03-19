import time
import numpy as np
from threading import Lock, Thread

from tminterface.client import Client
from tminterface.interface import TMInterface


class SimStateClient(Client):
    """
    Client for a TMInterface instance.
    Its only job is to get the simulation state that is used by the gym env for reward computation.
    """

    def __init__(self):
        super().__init__()
        self.sim_state = None
        self.action = None
        self.timer = time.time()
        self.restart = False
        

    def on_run_step(self, iface, _time: int):
        self.sim_state = iface.get_simulation_state()
        current_action = {
            'sim_clear_buffer': True,
            "steer":          int(np.clip(self.action[1]*65536, -65536, 65536)),  
            "gas":            int(np.clip(-self.action[0]*65536, -65536, 65536)),   
            }
        
        iface.set_input_state(**current_action)
        
        if self.restart:

            # iface.respawn()
            time.sleep(0)
            iface.execute_command("load_state")
            # iface.execute_command("warp 0000")
            
            self.restart = False

    def respawn(self):
        self.restart = True 


class ThreadedClient:
    """
    Allows to run the Client in a separate thread, so that the gym env can run in the main thread.
    """

    def __init__(self) -> None:
        self.iface = TMInterface()
        self.client = SimStateClient()
        self._client_thread = Thread(target=self.client_thread, daemon=True)
        self._lock = Lock()
        self.data = None
        self.action = [0, 0]
        self._client_thread.start()

    def client_thread(self):
        self.iface.register(self.client)
        while self.iface.running:
            time.sleep(0)
            self._lock.acquire()
            self.client.action = self.action
            self.data = self.client.sim_state
            self._lock.release()

if __name__ == "__main__":
    simthread = ThreadedClient()
    
    

