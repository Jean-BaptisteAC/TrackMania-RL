from TestBed import TestBed
from src.environment.TMNFEnv import TrackmaniaEnv
from stable_baselines3.common.env_checker import check_env 

from stable_baselines3 import PPO
import torch

if __name__ == "__main__":
    
    """ TRAIN AGENT """

    algorithm = "PPO"
    policy = "MlpPolicy"
    model_name = "PPO_test_log_interval"
    parameters_dict = {"action_space":"controller", "observation_space":"lidar"}
    save_interval = 10_000
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                    net_arch=[32, 32])
    seed=0

    testbed = TestBed(algorithm=algorithm, 
                      policy=policy,
                      model_name=model_name, 
                      parameters_dict=parameters_dict, 
                      save_interval=save_interval, 
                      policy_kwargs=policy_kwargs, 
                      seed=seed)
    
    testbed.train(200_000)