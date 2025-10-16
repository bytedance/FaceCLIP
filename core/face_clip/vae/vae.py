import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from typing import *
from core.util.common import get_obj_from_str


class BasicSD15VAE(nn.Module):

    base_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"

    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(BasicSD15VAE.base_model, subfolder='vae')

    def encode(self, image: torch.FloatTensor, return_dict: bool = True):
        latent = self.model.encode(image.to(self.model.device)).latent_dist.mode() * self.model.scaling_factor
        return latent

    def decode(self, latent: torch.FloatTensor):
        decoded_image = self.model.decode(latent / self.model.scaling_factor).sample.clip(-1, 1)
        return decoded_image

class BasicSDxlVAE(nn.Module):
    base_model = "stabilityai/stable-diffusion-xl-base-1.0"

    def __init__(self, **kwargs):
        super().__init__()
        precision = kwargs.get('precision', 'torch.float32')
        precision = get_obj_from_str(precision)
        if precision in [torch.float16, torch.bfloat16]:
            variant = "fp16"
        else:
            variant = "fp32"
        self.model = AutoencoderKL.from_pretrained(
            BasicSDxlVAE.base_model, 
            subfolder='vae', 
            torch_dtype=precision,
            variant=variant
        )

    def encode(self, image: torch.FloatTensor, return_dict: bool = True):
        latent = self.model.encode(image.to(self.model.device)).latent_dist.sample() * self.model.config.scaling_factor
        return latent

    def decode(self, latent: torch.FloatTensor):
        decoded_image = self.model.decode(latent / self.model.config.scaling_factor).sample
        return decoded_image

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