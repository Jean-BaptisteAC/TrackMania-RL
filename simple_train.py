from TestBed import TestBed
from src.environment.TMNFEnv import TrackmaniaEnv
from stable_baselines3.common.env_checker import check_env 

from stable_baselines3 import PPO
import torch

if __name__ == "__main__":
    
    """ TRAIN AGENT """

    # algorithm = "PPO"
    # model_name = "PPO_first_turn_continuous"
    # parameters_dict = {"action_space":"controller"}
    # save_interval = 10_000
    # policy_kwargs = dict(activation_fn=torch.nn.ReLU,
    #                 net_arch=[32, 32])
    # seed=0

    # testbed = TestBed(algorithm=algorithm, 
    #                   model_name=model_name, 
    #                   parameters_dict=parameters_dict, 
    #                   save_interval=save_interval, 
    #                   policy_kwargs=policy_kwargs, 
    #                   seed=seed)
    
    # testbed.train(200_000)
    
    """ LOAD AGENT """


    env = TrackmaniaEnv(action_space="controller")
    env.reset()
    model_type = "PPO"
    models_dir = "models/" + model_type
    model_watch_name = "PPO_first_turn_continuous"
    model_step = "80k"

    model_path = f"{models_dir}/{model_watch_name}/{model_step}"
    model_to_watch = PPO.load(model_path, env=env)
    
    while True:
        obs, _ = env.reset()
        done = False
        while not done:
            action, _states = model_to_watch.predict(obs)
            obs, reward, done, _, info = env.step(action)
            