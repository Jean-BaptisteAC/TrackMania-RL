import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_checker import check_env 

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

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space["image"].shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # self.linesight_cnn = nn.Sequential(
        #     nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 4), stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Flatten(),
        # )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten + 1, features_dim), nn.ReLU(),
            )

    def forward(self, observations: dict) -> th.Tensor:
        image = observations["image"]
        physics = observations["physics"]

        image_embedding = self.cnn(image)
        embedding = th.cat([image_embedding, physics], dim=1)

        return self.linear(embedding)


if __name__ == "__main__":
    
    """ TRAIN AGENT """

    algorithm = "PPO"
    policy = "MultiInputPolicy"
    model_name = "PPO_CNN+Velocity"
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
                      enable_log = True,
                      policy_kwargs=policy_kwargs, 
                      seed=seed)
    
    # testbed.train(200_000)

    print(testbed.model.policy)