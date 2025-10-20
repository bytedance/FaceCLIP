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
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import *
from core.face_t5.face_vit.vit import VisionTransformer
from core.face_t5.transformer.transformer import TransformerEncoderLayer, TransformerEncoder
from core.face_t5.transformer.fuse_modules import BiAttentionBlock
from core.util.common import instantiate_from_config, get_obj_from_str
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

def load_pretrained_fat5_xxl_model():
    import yaml
    from core.face_t5.flash_t5.model.configuration_flash_t5 import FlashT5Config
    from core.face_t5.flash_t5.model.modeling_flash_t5 import FlashT5ForConditionalGeneration
    with open("core/face_t5/flash_t5/config/t5-v1_1-xxl.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    config_hf_model = FlashT5Config.from_dict(config["model_args"])
    fat5_model = FlashT5ForConditionalGeneration(config=config_hf_model)
    return fat5_model

def load_pretrained_fat5_xxl_encoder():
    import yaml
    import copy
    from core.face_t5.flash_t5.model.configuration_flash_t5 import FlashT5Config
    from core.face_t5.flash_t5.model.modeling_flash_t5 import FlashT5Stack
    with open("core/face_t5/flash_t5/config/t5-v1_1-xxl.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)
    config = FlashT5Config.from_dict(config["model_args"])

    shared = nn.Embedding(config.vocab_size, config.d_model)

    encoder_config = copy.deepcopy(config)
    encoder_config.is_decoder = False
    encoder_config.use_cache = False
    encoder_config.is_encoder_decoder = False
    encoder = FlashT5Stack(encoder_config, shared)
    return encoder

class IdentityProjection(nn.Module):
    def __init__(self, seq_len, input_dim, output_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ln = nn.Linear(seq_len * input_dim, output_dim)

    def forward(self, x: torch.Tensor, normalize: bool = False) -> torch.Tensor:
        batch_size, seq_len, input_dim = x.shape
        x = x.reshape(batch_size, seq_len * input_dim)
        out = self.ln(x)
        out = F.normalize(out, dim=-1) if normalize else out
        return out


class TextFaceEmbeddingFuser(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_queries=300,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.0,
            activation="relu",
            normalize_before=False,
            return_intermediate_dec=True,
            query_dim=4,
            num_patterns=0,
            # for deformable encoder
            num_feature_levels=1,
            enc_n_points=4,
            dec_n_points=4,
            # init query
            learnable_tgt_init=True,
            # two stage
            two_stage_type="standard",
            embed_init_tgt=False,
            # for text
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_checkpoint=True,
            use_transformer_ckpt=True,
            use_text_cross_attention=True,
            text_dropout=0.1,
            fusion_dropout=0.1,
            fusion_droppath=0.0,
    ):
        super().__init__()
        self.num_feature_levels = num_feature_levels
        self.num_encoder_layers = num_encoder_layers
        self.num_unicoder_layers = num_unicoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_queries = num_queries
        assert query_dim == 4

        # choose encoder layer type
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            nhead=nhead
        )

        if use_text_enhancer:
            text_enhance_layer = TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead // 2,
                dim_feedforward=dim_feedforward // 2,
                dropout=text_dropout,
            )
        else:
            text_enhance_layer = None

        if use_fusion_layer:
            feature_fusion_layer = BiAttentionBlock(
                v_dim=d_model,
                l_dim=d_model,
                embed_dim=dim_feedforward // 2,
                num_heads=nhead // 2,
                dropout=fusion_dropout,
                drop_path=fusion_droppath,
            )
        else:
            feature_fusion_layer = None

        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        assert encoder_norm is None
        self.encoder = TransformerEncoder(
            encoder_layer,
            num_encoder_layers,
            d_model=d_model,
            num_queries=num_queries,
            text_enhance_layer=text_enhance_layer,
            feature_fusion_layer=feature_fusion_layer,
            use_checkpoint=use_checkpoint,
            use_transformer_ckpt=use_transformer_ckpt,
        )
        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries 
        self.num_patterns = num_patterns
        if not isinstance(num_patterns, int):
            Warning("num_patterns should be int but {}".format(type(num_patterns)))
            self.num_patterns = 0

        if num_feature_levels > 1:
            if self.num_encoder_layers > 0:
                self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
            else:
                self.level_embed = None

        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"
        self.embed_init_tgt = embed_init_tgt
        if (two_stage_type != "no" and embed_init_tgt) or (two_stage_type == "no"):
            self.tgt_embed = nn.Embedding(self.num_queries, d_model)
            nn.init.normal_(self.tgt_embed.weight.data)
        else:
            self.tgt_embed = None

        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries) 

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, text_embedding, text_pos_embedding, text_self_attn_mask, face_cls_embed, face_patch_embed,
                face_pos_embedding) -> torch.Tensor:
        bs = text_embedding.shape[0]
        dtype = text_embedding.dtype
        spatial_shapes = torch.tensor([[12, 12]], dtype=torch.long, device=face_patch_embed.device)
        valid_ratios = torch.tensor([[[1., 1.]]], dtype=dtype, device=face_patch_embed.device).repeat(
            text_embedding.shape[0], 1, 1)
        assert (face_patch_embed.shape[1] == spatial_shapes[0][0] * spatial_shapes[0][1])
        mask_flatten = torch.zeros(
            [face_patch_embed.shape[0], face_patch_embed.shape[1]], 
            dtype=torch.bool, 
            device=face_patch_embed.device
        )
        text_token_mask = torch.ones([text_embedding.shape[0], text_embedding.shape[1]], device=text_embedding.device,
                                     dtype=torch.bool)
        level_start_index = torch.tensor([0], device=text_embedding.device, dtype=torch.long)
        attn_mask = None
        memory, memory_text = self.encoder(
            face_patch_embed,
            pos=face_pos_embedding,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
            memory_text=text_embedding,
            text_attention_mask=~text_token_mask,
            pos_text=text_pos_embedding,
            text_self_attention_masks=text_self_attn_mask,
        )
        
        return memory_text

class FaceT5Encoder(nn.Module):

    def __init__(
            self,
            face_model_path: str = '',
            **kwargs
    ):
        super().__init__()
        precision = get_obj_from_str(kwargs.get('precision', 'torch.float32'))

        # Face Encoder
        self.face_encoder = VisionTransformer.from_pretrained(face_model_path).to(precision)

        # T5 Encoder
        self.t5_encoder = load_pretrained_fat5_xxl_encoder().to(precision).eval()
        self.context_length = self.t5_encoder.config.max_sequence_length

        self.face_patch_proj = nn.Linear(self.face_encoder.embed_dim, 4096)
        self.face_pos_proj = nn.Linear(self.face_encoder.embed_dim, 4096)
        self._register_text_pos_embedding(self.context_length, 4096)

        # Fuser
        self.fuser = TextFaceEmbeddingFuser(
            d_model=4096,
            nhead=8,
            num_queries=self.context_length,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_text_cross_attention=True,
            dim_feedforward=2048,
            dropout=0.0,
        ).to(precision)
        self._freeze_face_encoder()
        self._freeze_t5_encoder()


    def _register_text_pos_embedding(self, max_len, d_model):
        # Create a (max_len, d_model) matrix of positional encodings
        positional_embedding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the div_term (1 / 10000^(2i/d_model))
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                             (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices in the array; 2i
        positional_embedding[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices in the array; 2i+1
        positional_embedding[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension (1, max_len, d_model) and register as buffer
        positional_embedding = positional_embedding.unsqueeze(0)
        self.register_buffer('positional_embedding', positional_embedding)

    def _freeze_face_encoder(self):
        for param in self.face_encoder.parameters():
            param.requires_grad = False

    def _freeze_t5_encoder(self):
        for param in self.t5_encoder.parameters():
            param.requires_grad = False

    def encode_text(self, input_ids):
        output_dict = {}
        encoder_outputs = self.t5_encoder(
            input_ids=input_ids,
        )
        output_dict['text_embedding'] = encoder_outputs[0]
        return output_dict

    def forward_all(self, image, input_ids, device='cpu'):
        batch_size = len(input_ids)

        # face
        if isinstance(image, list):
            image_ = torch.stack(image, dim=0)
        else:
            image_ = image
        face_feature_ = self.face_encoder(
            F.interpolate(
                image_,
                size=(112, 112),
                mode='bilinear',
                align_corners=False
            ),
            dtype=image_.dtype
        )
        face_cls_embed, face_patch_embed = face_feature_['class_token'], face_feature_['patch_token']
        norm_face_cls_embed = face_cls_embed / face_cls_embed.norm(dim=-1, keepdim=True).to(face_cls_embed.dtype)

        face_pos_embedding = self.face_encoder.pos_embed.repeat(
            face_patch_embed.shape[0], 1, 1
        ).to(face_patch_embed.device)
        face_pos_embedding = self.face_pos_proj(face_pos_embedding)
        face_patch_embed = self.face_patch_proj(face_patch_embed)

        # text
        t5_output = self.encode_text(input_ids)
        text_embedding = t5_output['text_embedding']
        text_pos_embedding = self.positional_embedding.repeat(text_embedding.shape[0], 1, 1)

        # fuse
        fused_feature = self.fuser(
            text_embedding,
            text_pos_embedding,
            None,
            norm_face_cls_embed,
            face_patch_embed,
            face_pos_embedding
        )
        return fused_feature, norm_face_cls_embed, face_patch_embed

    def forward(self, aligned_face_image, face_exist_mask, input_ids, return_feat=False, device='cpu'):
        # text
        fused_feature, face_feat, face_patch_embed = self.forward_all(
            aligned_face_image, input_ids, device
        )

        if return_feat:
            return fused_feature, face_feat, face_patch_embed
        else:
            return fused_feature


@dataclass
class TextModelOutput:
    embeddings: torch.Tensor
    masks: torch.Tensor
    pooled: List

class FaceT5FluxWrapper(nn.Module):

    base_model = "black-forest-labs/FLUX.1-dev"

    def __init__(self, precision='torch.bfloat16'):
        super().__init__()
        precision = get_obj_from_str(precision)
        self._init_model_from_ckpt(precision)

    def _init_model_from_ckpt(self, precision=torch.bfloat16):
        from transformers import CLIPTokenizer, CLIPTextModel
        config = OmegaConf.load('configs/face_t5_xxl.yaml')
        # load model
        self.tokenizer_1 = CLIPTokenizer.from_pretrained(FaceT5FluxWrapper.base_model, subfolder="tokenizer")
        self.encoder_1 = CLIPTextModel.from_pretrained(FaceT5FluxWrapper.base_model, subfolder="text_encoder").to(precision)
        self.tokenizer_2 = AutoTokenizer.from_pretrained(FaceT5FluxWrapper.base_model, subfolder="tokenizer_2")
        self.encoder_2 = instantiate_from_config(config.model.params.encoder_config)
        self.encoder_2.to(precision)
        self._freeze()

    def _freeze(self):
        for param in self.encoder_1.parameters():
            param.requires_grad = False
        for param in self.encoder_2.parameters():
            param.requires_grad = False

    def forward(self, aligned_face_image, face_exist_mask, text, device='cpu'):
        input_ids_1 = self.tokenizer_1(
            text,
            max_length=self.tokenizer_1.model_max_length,
            padding="max_length",
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt"
        ).input_ids.to(device)

        input_ids_2 = self.tokenizer_2(
            text,
            max_length=self.tokenizer_2.model_max_length,
            padding="max_length",
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt"
        ).input_ids.to(device)

        clip_prompt_embeds = self.encoder_1(input_ids_1, output_hidden_states=False)
        pooled_prompt_embeds = clip_prompt_embeds.pooler_output

        mixed_feature, face_feature, _ = \
            self.encoder_2(aligned_face_image, face_exist_mask, input_ids_2, return_feat=True, device=device)

        return TextModelOutput(
            embeddings=mixed_feature,
            masks=None,
            pooled=[pooled_prompt_embeds]
        )