import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from typing import *
from core.util.common import get_obj_from_str

class BasicFluxDevVAE(nn.Module):
    base_model = "black-forest-labs/FLUX.1-dev"

    def __init__(self, **kwargs):
        super().__init__()
        precision = kwargs.get('precision', 'torch.float32')
        precision = get_obj_from_str(precision)
        self.model = AutoencoderKL.from_pretrained(BasicFluxDevVAE.base_model, subfolder="vae").to(precision)

    def encode(self, image: torch.FloatTensor, return_dict: bool = True):
        latents = self.model.encode(image.to(self.model.device)).latent_dist.sample()
        latents = (latents - self.model.config.shift_factor) * self.model.config.scaling_factor
        return latents.to(self.model.dtype)

    def decode(self, latents: torch.FloatTensor):
        decode_images = self.model.decode(latents.to(self.model.device) / self.model.config.scaling_factor  + self.model.config.shift_factor).sample
        return decode_images

class BasicSD3VAE(nn.Module):
    base_model = "stabilityai/stable-diffusion-3-medium-diffusers"

    def __init__(self, **kwargs):
        super().__init__()
        precision = kwargs.get('precision', 'torch.float32')
        precision = get_obj_from_str(precision)
        self.model = AutoencoderKL.from_pretrained(BasicSD3VAE.base_model, subfolder="vae").to(precision)

    def encode(self, image: torch.FloatTensor, return_dict: bool = True):
        latents = self.model.encode(image.to(self.model.device)).latent_dist.sample()
        latents = (latents - self.model.config.shift_factor) * self.model.config.scaling_factor
        return latents.to(self.model.dtype)

    def decode(self, latents: torch.FloatTensor):
        decode_images = self.model.decode(latents.to(self.model.device) / self.model.config.scaling_factor + self.model.config.shift_factor).sample
        return decode_images