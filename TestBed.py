
from src.environment.TMNFEnv import TrackmaniaEnv
from stable_baselines3 import PPO, SAC
from tqdm.auto import tqdm
import torch as th
import numpy as np
import os

def return_model(algorithm):
    if algorithm == "PPO":
        return PPO
    if algorithm == "SAC":
        return SAC
    
class TestBed:
    def __init__(self, algorithm, policy, model_name, parameters_dict, save_interval, **kwargs):
        self.algorithm = algorithm
        self.policy = policy
        self.model_name = model_name
        self.save_interval = save_interval
        self.parameters_dict = parameters_dict
        self.total_timestep = 0
        
        SB3_arguments = {}

        if "policy_kwargs" in kwargs:
            SB3_arguments["policy_kwargs"] = kwargs["policy_kwargs"]
        if "learning_rate" in kwargs:
            SB3_arguments["learning_rate"] = kwargs["learning_rate"]
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
                                             n_steps=4096,
                                             tensorboard_log=self.logdir,
                                             **SB3_arguments)
        
        self.models_dir = "models/" + self.algorithm + "/" + self.model_name
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
        self.step = 0
        
    def load_agent(self, model_path):
        self.baseline = model_path
        self.model = return_model(self.algorithm).load(model_path, env=self.env)
        
    
    def train(self, time_steps):
        
        self.total_timestep += time_steps
        
        while self.step < self.total_timestep:
            self.model.learn(total_timesteps=self.save_interval, 
                             reset_num_timesteps=False, 
                             tb_log_name=self.model_name)
            self.step += self.save_interval

            model_step = str(int(self.step/1000)) + "k"
            self.model.save(f"{self.models_dir}/{model_step}")