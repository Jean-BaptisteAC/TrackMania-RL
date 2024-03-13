import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from TestBed import TestBed
from src.environment.TMNFEnv import TrackmaniaEnv

from stable_baselines3 import PPO

# Simple CNN taken from the SB3 custom Policy example
class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


if __name__ == "__main__":
    
    """ TRAIN AGENT """

    algorithm = "PPO"
    policy = "CnnPolicy"
    model_name = "PPO_first_CNN"
    parameters_dict = {"action_space":"controller", "observation_space":"image"}
    save_interval = 10_000
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
    )   
    seed=0

    testbed = TestBed(algorithm=algorithm,
                      policy=policy,
                      model_name=model_name, 
                      parameters_dict=parameters_dict, 
                      save_interval=save_interval, 
                      policy_kwargs=policy_kwargs, 
                      seed=seed)
    
    testbed.train(200_000)