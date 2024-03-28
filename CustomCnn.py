from typing import Callable, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch as th
from torch import nn

from TestBed import TestBed

# Simple CNN taken from the SB3 custom Policy example
class CNN_Extractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.

    
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space["image"].shape[0]

        self.linesight_cnn = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 4), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )

        self.nature_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.linesight_cnn(
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        physics_shape = observation_space["physics"].shape[0]
        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, features_dim - physics_shape), 
            nn.ReLU()
            )
    
    def forward(self, observation: spaces.Dict) -> Tuple[th.Tensor, th.Tensor]:
        image_embedding = self.cnn_head(self.linesight_cnn(observation["image"]))
        embedding = th.cat([image_embedding, observation["physics"]], dim=1)
        return embedding


if __name__ == "__main__":

    """ TRAIN AGENT """

    algorithm = "PPO"
    model_name = "PPO_Training_Flat_Dataset_env_fix"
    parameters_dict = {"observation_space":"image", "dimension_reduction":6}
    save_interval = 12_288
    policy_kwargs = dict(
        features_extractor_class=CNN_Extractor,
        features_extractor_kwargs=dict(features_dim=128),
        activation_fn=th.nn.ReLU, 
        net_arch=[128, 128],
    )   
    seed=0
    # buffer_size = 50_000
    # train_freq  = (1_000, "step")

    testbed = TestBed(algorithm=algorithm,
                      policy="MultiInputPolicy",
                      model_name=model_name, 
                      parameters_dict=parameters_dict, 
                      save_interval=save_interval,
                      policy_kwargs=policy_kwargs, 
                      seed=seed)
    
    # agent_path = "models/PPO/PPO_Training_Flat_Dataset/390k"
    # testbed.load_agent(model_path=agent_path, step=390_000)
    
    testbed.train(1_000_000)