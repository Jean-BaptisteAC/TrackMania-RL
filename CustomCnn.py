from typing import Callable, Tuple

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch as th
from torch import nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights

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
        self.input_size = observation_space["image"]

        self.cnn_upgraded = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=32, kernel_size=(5, 5), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )

        self.cnn_base = nn.Sequential(
            nn.Conv2d(in_channels=n_input_channels, out_channels=16, kernel_size=(5, 5), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3, 3), stride=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )



        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn_base(
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        physics_shape = observation_space["physics"].shape[0]
        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, features_dim - physics_shape), 
            nn.ReLU()
            )
    
    def forward(self, observation: spaces.Dict) -> Tuple[th.Tensor, th.Tensor]:
        image_embedding = self.cnn_head(self.cnn_base(observation["image"]))
        embedding = th.cat([image_embedding, observation["physics"]], dim=1)
        return embedding
    

class CNN_Extractor_Resnet(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.

        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        
        n_input_channels = observation_space["image"].shape[0]
        self.input_size = observation_space["image"]

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # RESNET
        self.resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        
        for name, param in self.resnet18.named_parameters():
            if "fc" in name:  # Unfreeze the final classification layer
                param.requires_grad = True
            else:
                param.requires_grad = False

        # NATURE CNN
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
            n_flatten = self.resnet18(
                th.as_tensor(observation_space["image"].sample()[None]).float()
            ).shape[1]

        physics_shape = observation_space["physics"].shape[0]
        self.cnn_head = nn.Sequential(
            nn.Linear(n_flatten, features_dim - physics_shape), 
            nn.ReLU()
            )
    
    def forward(self, observation: spaces.Dict) -> Tuple[th.Tensor, th.Tensor]:
        
        image = observation["image"]
        preprocessed_image = self.preprocess(image)
        image_embedding = self.cnn_head(self.resnet18(preprocessed_image))
        embedding = th.cat([image_embedding, observation["physics"]], dim=1)
        return embedding

if __name__ == "__main__":

    """ TRAIN AGENT """

    algorithm = "PPO"
    model_name = "PPO_resnet"

    parameters_dict = {"observation_space":"image", 
                       "dimension_reduction":6,
                       "training_track":"A03",
                       "training_mode":"time_optimization",
                       "is_testing":False}
    
    save_interval = 12_288
    policy_kwargs = dict(
        features_extractor_class=CNN_Extractor_Resnet,
        features_extractor_kwargs=dict(features_dim=256),
        activation_fn=th.nn.Tanh, 
        net_arch=[256, 256],
    )   
    seed=0
    learning_rate = 1e-4
    use_sde = True
    n_steps = 2048

    testbed = TestBed(algorithm=algorithm,
                      policy="MultiInputPolicy",
                      model_name=model_name,
                      parameters_dict=parameters_dict,
                      save_interval=save_interval,
                      policy_kwargs=policy_kwargs,
                      seed=seed,
                      learning_rate=learning_rate, 
                      use_sde=use_sde, 
                      n_steps=n_steps)
    
    # print(testbed.model.policy)
    
    # agent_path = "models/PPO/PPO_Training_Dataset_Tech_2_small_CNN/1277k"
    # testbed.load_agent(model_path=agent_path, step=1_277_000, parameters_to_change={})

    testbed.train(1_000_000)
    