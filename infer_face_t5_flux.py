# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor
from core.face_t5.dit.flux import FluxModel
from core.face_t5.face_t5 import FaceT5FluxWrapper
from diffusers.pipelines.flux.pipeline_flux import FluxPipeline
from core.face_t5.vae.vae import BasicFluxDevVAE
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

repo_id = 'ByteDance/FaceCLIP'

def read_image(image_path: str):
    return Image.open(image_path).convert('RGB')

def init_model_from_ckpt(device='cuda'):
    vae = BasicFluxDevVAE().model

    # hf_hub_download(
    #     repo_id,
    #     filename='model/tokenizer',
    #     local_dir='./'
    # )

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir="./",
        allow_patterns=["model/tokenizer/*"]
    )


    # text_image_encoder
    hf_hub_download(
        repo_id,
        filename='model/face_t5_encoder.pt',
        local_dir='./'
    )
    text_image_encoder = FaceT5FluxWrapper()
    state_dict = torch.load('model/face_t5_encoder.pt', map_location='cpu')

    text_image_encoder.load_state_dict(state_dict)
    text_image_encoder.to(device)
    text_image_encoder = text_image_encoder.eval()

    # dit
    hf_hub_download(
        repo_id,
        filename='model/dit.pt',
        local_dir='./'
    )
    # dit
    dit = FluxModel.from_pretrained('black-forest-labs/FLUX.1-dev').to(torch.bfloat16)
    dit_state_dict = torch.load('model/dit.pt')
    dit.load_state_dict(dit_state_dict)
    dit.eval()
    dit.to(device)
    del dit_state_dict

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        'black-forest-labs/FLUX.1-dev',
        subfolder="scheduler"
    )
    tokenizer = text_image_encoder.tokenizer_1
    text_encoder = text_image_encoder.encoder_1
    tokenizer_2 = text_image_encoder.tokenizer_2
    text_encoder_2 = text_image_encoder.encoder_2.t5_encoder

    pipeline = FluxPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        transformer=dit,
        scheduler=noise_scheduler,
    )

    pipeline = pipeline.to(device)
    model_dict = {
        'pipeline': pipeline,
        'text_image_encoder': text_image_encoder
    }
    return model_dict

def do_inference(model, aligned_face_image, text, num_images):
    pipeline = model['pipeline']
    text_image_encoder = model['text_image_encoder']
    aligned_face_image = [m.to(torch.bfloat16) for m in aligned_face_image]
    zero_face_image = [torch.zeros_like(im) for im in aligned_face_image]
    null_text = [
        "digital painting, 3d render, CGI, cartoon, anime, unreal, stylized, overexposed, glossy skin, "
        "wearing mask, looking away from the camera, wearing glasses, wearing sunglasses, "
        "nude, worst quality, low quality, normal quality, nsfw, abstract, glitch, deformed, "
        "mutated, ugly, disfigured, text, watermark, bad hands, error, jpeg artifacts, blurry, "
        "missing fingers, poorly drawn face, ugly eyes, imperfect eyes, deformed pupils, deformed iris, bad anatomy"
    ] * len(text)

    with torch.autocast(device_type=aligned_face_image[0].device.type, dtype=torch.bfloat16):
        face_exist_mask = torch.ones(len(aligned_face_image), dtype=torch.bool, device=aligned_face_image[0].device)
        prompt_embeds = \
            text_image_encoder(
            aligned_face_image, face_exist_mask, text, device=aligned_face_image[0].device
        )
        negative_prompt_embeds = \
            text_image_encoder(
            zero_face_image, ~face_exist_mask, null_text, device=aligned_face_image[0].device
        )

    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        res = pipeline(
            prompt_embeds=prompt_embeds.embeddings,
            negative_prompt_embeds=negative_prompt_embeds.embeddings,
            pooled_prompt_embeds=prompt_embeds.pooled[-1],
            negative_pooled_prompt_embeds=negative_prompt_embeds.pooled[-1],
            num_inference_steps=50,
            guidance_scale=1.0,
            true_cfg_scale=3.5,
            num_images_per_prompt=num_images,
            height=1152,
            width=864,
        ).images[0]
    return res

def save_image(path: str, name: str, image: Image):
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, name)
    image.save(path)

if __name__ == '__main__':
    device = 'cuda'
    model = init_model_from_ckpt()

    # prepare input
    face_path = "asset/0001_female.png"
    face_image_pil = Image.open(face_path).convert('RGB')
    face_transforms = ToTensor()
    face_image_tensor = face_transforms(face_image_pil)
    face_image = [face_image_tensor.to(device)]

    # text
    text = 'A woman wearing a blue denim shirt and khaki pants, on a rocky desert plateau'
    res_image = do_inference(model, face_image, [text], num_images=1)
    os.makedirs("output_image", exist_ok=True)
    save_image("output_image", f"{text.replace(' ', '_')}_.png", res_image)

