from stable_baselines3.common.env_checker import check_env
from env.TMNFEnv import TrackmaniaEnv
import torch

from stable_baselines3 import PPO

env = TrackmaniaEnv()
# check_env(env)

if __name__ == "__main__":
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                     net_arch=[16, 16])

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    
    model.learn(total_timesteps=50_000)
    model.save("models/PPO_16_50k")

