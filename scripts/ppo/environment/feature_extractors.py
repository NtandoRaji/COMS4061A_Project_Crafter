import os
import torch as T
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
from scripts.ppo.vae.vae_model import VariationalAutoEncoder


class CrafterLatentFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, latent_dim: int, model_path: str, device: str = "cpu"):
        super().__init__(observation_space, features_dim=latent_dim)
        self.device = device
        
        self.encoder = self.load_encoder(observation_space, latent_dim, model_path, device)

        for param in self.encoder.parameters():
            param.requires_grad = False

    def load_encoder(self, observation_space: Box, latent_dim: int, model_path: str, device: str = "cpu"):
        vae_model = VariationalAutoEncoder(observation_space.shape, latent_dim, device)
        vae_model.load(model_path)
        vae_model.eval()
        return vae_model.encoder

    def forward(self, observations: T.Tensor) -> T.Tensor:
        x = observations
        _, mu, _ = self.encoder(x)
        return mu
