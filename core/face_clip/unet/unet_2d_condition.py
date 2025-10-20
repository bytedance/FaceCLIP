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


import torch
import math
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel, UNet2DConditionOutput
from typing import Any, Dict, Optional, Tuple, List, Union, Callable


def _reset_unet_cross_attn(unet):
    def reset_parameters(module):
        torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
        if module.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(module.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(module.bias, -bound, bound)

    for name, module in unet.named_modules():
        if name.split('.')[-1].startswith('attn2'):
            reset_parameters(module.to_k)
            reset_parameters(module.to_q)
            reset_parameters(module.to_v)
            reset_parameters(module.to_out[0])


def _add_loras(model, r=8, lora_alpha=8, lora_dropout=0.1, use_rslora=False):
    target_modules = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    peft_config = LoraConfig(
        target_modules=target_modules,
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_rslora=use_rslora
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model

class MyUNet2DConditionModel(UNet2DConditionModel):

    @staticmethod
    def from_pretrained(model_path: str, use_safetensors: bool = True, **kwargs):
        precision = kwargs.get('precision', torch.float32)
        train_lora = kwargs.get('train_lora', False)
        lora_rank = kwargs.get('lora_rank', 8)
        lora_alpha = kwargs.get('lora_alpha', lora_rank)
        lora_dropout = kwargs.get('lora_dropout', 0.1)
        use_rslora = kwargs.get('use_rslora', False)
        if precision in [torch.float16, torch.bfloat16]:
            variant = "fp16"
        else:
            variant = "fp32"

        model = UNet2DConditionModel.from_pretrained(
            model_path, use_safetensors=use_safetensors, 
            torch_dtype=precision, variant=variant, **kwargs
        )

        if train_lora:
            model = _add_loras(
                model,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=use_rslora
            )
        return model.train()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, *args, **kwargs) -> Union[UNet2DConditionOutput, Tuple]:
        return super().forward(*args, **kwargs)
