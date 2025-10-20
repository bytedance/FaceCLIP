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
import re
# from flash_attn import flash_attn_func
from typing import *
# from diffusers.models.attention_processor import Attention
from diffusers.models import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from peft import LoraConfig, get_peft_model

class FluxModel(FluxTransformer2DModel):

    @staticmethod
    def from_pretrained(model_path: str, **kwargs):
        arch_only = kwargs.get('arch_only', False)
        gradient_checkpointing = kwargs.get('gradient_checkpointing', False)
        train_lora = kwargs.get('train_lora', False)
        lora_rank = kwargs.get('lora_rank', 8)
        lora_alpha = kwargs.get('lora_alpha', lora_rank)
        lora_dropout = kwargs.get('lora_dropout', 0.1)
        use_rslora = kwargs.get('use_rslora', False)
        precision = kwargs.get('precision', torch.float32)
        use_flash_attention = kwargs.get('use_flash_attention', False)
        # assert arch_only != ('lora' in kwargs)

        if arch_only:
            # Only create a scratch model
            transformer_config = FluxTransformer2DModel.load_config(model_path, subfolder="transformer")
            model = FluxTransformer2DModel.from_config(transformer_config).train()
        else:
            model = FluxTransformer2DModel.from_pretrained(model_path, subfolder="transformer").train()

        if gradient_checkpointing:
            model.enable_gradient_checkpointing()
            
        def _partial_freeze(model):
            for name, param in model.named_parameters():
                if name.startswith('context_embedder'):
                    param.requires_grad = True
                elif name.startswith('time_text_embed'):
                    param.requires_grad = True
                elif name.startswith('transformer_blocks'):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            return model

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

        # def apply_flash_attention(model):
        #     for m in model.modules():
        #         if isinstance(m, Attention):
        #             m.set_processor(FAFluxAttentionProcessor())
        #     return model

        # model = _reset_parameters(model)
        # model = _partial_freeze(model)
        if train_lora:
            model = _add_loras(
                model,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                use_rslora=use_rslora
            )
        # if use_flash_attention:
        #     model = apply_flash_attention(model)
        return model.to(precision)


    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            return_dict=False
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        return super().forward(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=return_dict
        )

