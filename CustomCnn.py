from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO
from gymnasium import spaces
import torch as th
from torch import nn

from src.environment.TMNFEnv import TrackmaniaEnv
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
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

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

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
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


class CustomPolicyNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """
    def __init__(
        self,
        feature_dim: int = 64,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64
    ):
        super().__init__()
        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.hiddel_dim = 64
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, self.hiddel_dim),
            nn.ReLU(),
            nn.Linear(self.hiddel_dim, self.latent_dim_pi), 
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, self.hiddel_dim),
            nn.ReLU(),
            nn.Linear(self.hiddel_dim, self.latent_dim_vf), 
        )

    def forward(self, features: spaces.Dict) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``

            On that example only the CNN and its head is shared between actor and critic
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomPolicyNetwork()


if __name__ == "__main__":

    """ TRAIN AGENT """

    algorithm = "PPO"
    model_name = "PPO_CNN+lateral+Soft_speed+Asymmetry"
    parameters_dict = {"action_space":"controller", "observation_space":"image"}
    save_interval = 10_000
    policy_kwargs = dict(
        features_extractor_class=CNN_Extractor,
        features_extractor_kwargs=dict(features_dim=64),
    )   
    seed=0

    testbed = TestBed(algorithm=algorithm,
                      policy=CustomActorCriticPolicy,
                      model_name=model_name, 
                      parameters_dict=parameters_dict, 
                      save_interval=save_interval,
                      policy_kwargs=policy_kwargs, 
                      seed=seed)
    
    testbed.train(200_000)