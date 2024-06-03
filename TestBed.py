
from src.environment.TMNFEnv import TrackmaniaEnv
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm
import torch as th
import numpy as np
import os

def return_model(algorithm):
    if algorithm == "PPO":
        return PPO
    if algorithm == "SAC":
        return SAC
    if algorithm == "TD3":
        return TD3
    
class RecordingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.time_recording = []
        self.distance_recording = []

    def _on_step(self) -> bool:
        value = self.locals['infos'][0]["checkpoint_time"]

        if value is not False:
            self.time_recording.append(value)
        
        if self.num_timesteps % 500 == 0:
            if len(self.time_recording) >= 1:
                recorded_value = np.array(self.time_recording).mean()
                self.logger.record("lap_time_mean", recorded_value)
                self.logger.dump(self.num_timesteps)
            self.time_recording = []

        value = self.locals['infos'][0]["total_distance"]

        if value is not False:
            self.distance_recording.append(value)
        
        if self.num_timesteps % 500 == 0:
            if len(self.distance_recording) >= 1:
                recorded_value = np.array(self.distance_recording).mean()
                self.logger.record("total_distance_mean", recorded_value)
                self.logger.dump(self.num_timesteps)
            self.distance_recording = []

        return True
    
class TestBed:
    def __init__(self, algorithm, policy, model_name, parameters_dict, save_interval, **kwargs):
        self.algorithm = algorithm
        self.policy = policy
        self.model_name = model_name
        self.parameters_dict = parameters_dict
        self.save_interval = save_interval
        self.total_timestep = 0
        
        SB3_arguments = {}

        if "policy_kwargs" in kwargs:
            SB3_arguments["policy_kwargs"] = kwargs["policy_kwargs"]
        if "learning_rate" in kwargs:
            SB3_arguments["learning_rate"] = kwargs["learning_rate"]
        if "buffer_size" in kwargs:
            SB3_arguments["buffer_size"] = kwargs["buffer_size"]
        if "train_freq" in kwargs:
            SB3_arguments["train_freq"] = kwargs["train_freq"]
        if "use_sde " in kwargs:
            SB3_arguments["use_sde "] = kwargs["use_sde "]
        if "n_steps" in kwargs:
            SB3_arguments["n_steps"] = kwargs["n_steps"]
        if "seed" in kwargs:
            SB3_arguments["seed"] = kwargs["seed"]
        else:
            self.seed = np.random.randint(0, high=100_000, dtype=int)
            SB3_arguments["seed"] = self.seed
        
        self.env = TrackmaniaEnv(**self.parameters_dict)
        
        self.logdir = "logs"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        
        self.model = return_model(algorithm)(self.policy,
                                             self.env,
                                             verbose=1,
                                             tensorboard_log=self.logdir,
                                             **SB3_arguments)
        
        self.models_dir = "models/" + self.algorithm + "/" + self.model_name
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
        self.step = 0
        
    def load_agent(self, model_path, step, parameters_to_change):
        self.baseline = model_path
        self.model = return_model(self.algorithm).load(model_path, 
                                                       env=self.env, 
                                                       custom_objects=parameters_to_change)
        self.step = step
        self.total_timestep = step
        
    
    def train(self, time_steps):
        
        self.total_timestep += time_steps
        
        while self.step < self.total_timestep:

            self.model.learn(total_timesteps=self.save_interval, 
                            reset_num_timesteps=False, 
                            log_interval = 1,
                            tb_log_name=self.model_name, 
                            callback=RecordingCallback())

            self.step += self.save_interval

            model_step = str(int(self.step/1000)) + "k"
            self.model.save(f"{self.models_dir}/{model_step}")
