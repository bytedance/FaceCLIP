import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open_clip
from dataclasses import dataclass
from core.util.common import get_obj_from_str
from typing import *
from core.face_clip.open_clip.model import _build_text_tower, _build_vision_tower
from core.face_clip.open_clip.transformer import  VisionTransformer,  text_global_pool
from core.face_clip.open_clip.factory import get_model_config, get_tokenizer
from core.face_clip.face_vit.vit import VisionTransformer
from groundingdino.transformer import DeformableTransformerEncoderLayer, \
    TransformerEncoderLayer, TransformerEncoder, DeformableTransformerDecoderLayer, \
    TransformerDecoder2
from groundingdino.fuse_modules import BiAttentionBlock
from groundingdino.ms_deform_attn import MultiScaleDeformableAttention as MSDeformAttn

from omegaconf import OmegaConf
from core.util.common import instantiate_from_config
from transformers import CLIPTokenizerFast



class TextProjection(nn.Module):
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
            two_stage_type="standard",  # ['no', 'standard', 'early', 'combine', 'enceachlayer', 'enclayer1']
            embed_init_tgt=False,
            # for text
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_checkpoint=False,
            use_transformer_ckpt=False,
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
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points
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

        """
        # choose decoder layer type
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            num_feature_levels,
            nhead,
            dec_n_points,
            use_text_cross_attention=use_text_cross_attention,
        )

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder2(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec,
            d_model=d_model,
            query_dim=query_dim,
            num_feature_levels=num_feature_levels,
        )
        """

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries  # useful for single stage model only
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

        # for two stage
        self.two_stage_type = two_stage_type
        assert two_stage_type in ["no", "standard"], "unknown param {} of two_stage_type".format(
            two_stage_type
        )
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.two_stage_wh_embedding = None

        if two_stage_type == "no":
            self.init_ref_points(num_queries)  # init self.refpoint_embed

        """
        self.enc_out_class_embed = ContrastiveEmbed(max_text_len=num_queries + 2)
        self.enc_out_bbox_embed = MLP(768, 256, 4, 3)
        """
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

    def init_ref_points(self, use_num_queries):
        self.refpoint_embed = nn.Embedding(use_num_queries, 4)

    def forward(self, text_embedding, text_pos_embedding, text_self_attn_mask, face_cls_embed, face_patch_embed,
                face_pos_embedding) -> torch.Tensor:
        bs = text_embedding.shape[0]
        spatial_shapes = torch.tensor([[12, 12]], dtype=torch.long, device=face_patch_embed.device)
        valid_ratios = torch.tensor([[[1., 1.]]], dtype=torch.float, device=face_patch_embed.device).repeat(
            text_embedding.shape[0], 1, 1)
        assert (face_patch_embed.shape[1] == spatial_shapes[0][0] * spatial_shapes[0][1])
        mask_flatten = torch.zeros([face_patch_embed.shape[0], face_patch_embed.shape[1]], dtype=torch.bool,
                                   device=face_patch_embed.device)
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
            # we ~ the mask . False means use the token; True means pad the token
            # position_ids=torch.zeros([text_embedding.shape[0], text_embedding.shape[1]], device=text_embedding.device,
            #                          dtype=torch.long),
            pos_text=text_pos_embedding,
            text_self_attention_masks=text_self_attn_mask,
        )
        
        # only return the fuse feature
        return memory_text
        """
        text_dict = {}
        text_dict["encoded_text"] = memory_text
        text_dict["text_token_mask"] = text_token_mask

        output_memory, output_proposals = gen_encoder_output_proposals(
                memory, mask_flatten, spatial_shapes
            )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        if text_dict is not None:
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory, text_dict)
        else:
            enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)

        topk_logits = enc_outputs_class_unselected.max(-1)[0]
        # enc_outputs_coord_unselected = (
        #     self.enc_out_bbox_embed(output_memory) + output_proposals

        # we do not predict the box coord but use the predefined output_proposals
        enc_outputs_coord_unselected = output_proposals
        # )  # (bs, \sum{hw}, 4) unsigmoid
        # topk = self.num_queries

        # topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq

        # gather boxes
        # refpoint_embed_undetach = torch.gather(
        #     enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        # )  # unsigmoid
        refpoint_embed_undetach = enc_outputs_coord_unselected
        refpoint_embed_ = refpoint_embed_undetach.detach()
        # init_box_proposal = torch.gather(
        #     output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        # ).sigmoid()  # sigmoid

        # gather tgt
        # tgt_undetach = torch.gather(
        #     output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        # )
        # if self.embed_init_tgt:
        #     tgt_ = (
        #         self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        #     )  # nq, bs, d_model
        # else:
        #     tgt_ = tgt_undetach.detach()

        # # import pdb; pdb.set_trace()
        # if refpoint_embed is not None:
        #     refpoint_embed = torch.cat([refpoint_embed, refpoint_embed_], dim=1)
        #     tgt = torch.cat([tgt, tgt_], dim=1)
        # else:
        #     refpoint_embed, tgt = refpoint_embed_, tgt_
        
        refpoint_embed = refpoint_embed_

        hs = self.decoder(
            tgt=output_memory.transpose(0, 1),
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=face_pos_embedding.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=attn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
            # we ~ the mask . False means use the token; True means pad the token
        )
        return hs
        """

class TextImageBranch(nn.Module):

    def __init__(
            self,
            text_embed_dim: int,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            cast_dtype: Optional[torch.dtype] = None,
            text_model_name: str = '',
            face_model_path: str = '',
            pretrained: str = 'metaclip_fullcc'
    ):
        super().__init__()

        self.cast_dtype =  get_obj_from_str(cast_dtype)
        text_model_config = get_model_config(text_model_name)['text_cfg']
        quick_gelu = get_model_config(text_model_name).get('quick_gelu', False)
        text = _build_text_tower(text_embed_dim, text_model_config, quick_gelu, cast_dtype)
        # text, _, _ = open_clip.create_model_and_transforms(text_model_name, pretrained=pretrained)
        self.tokenizer = get_tokenizer(text_model_name)
        self.transformer = text.transformer
        self.context_length = text.context_length
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        # self.text_pool_type = text.text_pool_type
        self.text_pool_type = text.pool_type
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones([]) * init_logit_bias)
        else:
            self.logit_bias = None
        self.face_encoder = VisionTransformer.from_pretrained(face_model_path)
        self.proj1 = nn.Linear(self.face_encoder.num_classes, text_embed_dim)
        self.face_patch_proj = nn.Linear(self.face_encoder.embed_dim, text_embed_dim)
        self.face_pos_proj = nn.Linear(self.face_encoder.embed_dim, text_embed_dim)
        self.text_embed_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.text_pos_proj = nn.Linear(text_embed_dim, text_embed_dim)
        self.fuser = TextFaceEmbeddingFuser(
            d_model=text_embed_dim,
            nhead=8,
            num_queries=text.context_length,
            num_encoder_layers=6,
            num_unicoder_layers=0,
            num_decoder_layers=6,
            use_text_enhancer=True,
            use_fusion_layer=True,
            use_text_cross_attention=True,
            dim_feedforward=2048,
            dropout=0.0,
        )
        self._init_text_tower_from_pretrained(text_model_name, pretrained)
        self._freeze_face_encoder()
        self._freeze_text_modules()

    def _init_text_tower_from_pretrained(self, text_model_name, pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(text_model_name, pretrained=pretrained)
        self.transformer.load_state_dict(model.transformer.state_dict())
        self.token_embedding.load_state_dict(model.token_embedding.state_dict())
        self.positional_embedding = model.positional_embedding
        self.ln_final.load_state_dict(model.ln_final.state_dict())
        self.text_projection = model.text_projection
        self.text_pool_type = model.text_pool_type

    def _freeze_face_encoder(self):
        for param in self.face_encoder.parameters():
            param.requires_grad = False

    def _freeze_text_modules(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.ln_final.parameters():
            param.requires_grad = False
        for param in self.token_embedding.parameters():
            param.requires_grad = False
        self.positional_embedding.requires_grad = False
        self.text_projection.requires_grad = False
    
    def forward_clip_text(self, input_ids, normalize: bool = False):
        output_dict = {}
        cast_dtype = self.cast_dtype
        
        x = self.token_embedding(input_ids).to(cast_dtype)
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        output_dict['text_embedding'] = x
        output_dict['attn_mask'] = self.attn_mask
        x, _ = text_global_pool(x, input_ids, self.text_pool_type)
        output_dict['pooled'] = x
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        output_dict['text_feature'] = F.normalize(x, dim=-1) if normalize else x
        return output_dict


    def encode_text(self, input_ids, identity_text_mask, image_token_id, face_token, normalize: bool = False):
        output_dict = {}
        cast_dtype = self.cast_dtype

        # if identity_input_ids is not None:
        #     identity_len = (identity_input_ids != 0).sum(dim=-1)[0] - 1
        #     # [batch_size, n_ctx, d_model]
        #     x = self.token_embedding(input_ids).to(cast_dtype)
        #     id_insert_mask = (identity_input_ids == image_token_id)
        #     id_insert_mask[:, identity_len:] = False
        #     x[id_insert_mask] = face_token[(identity_input_ids == image_token_id).any(dim=-1)]
        # else:
        #     x = self.token_embedding(input_ids).to(cast_dtype)

        # identity_len = (identity_input_ids != 0).sum(dim=-1)[0] - 1
        # [batch_size, n_ctx, d_model]
        x = self.token_embedding(input_ids).to(cast_dtype)
        id_insert_mask = (input_ids == image_token_id) & identity_text_mask
        # id_insert_mask[:, identity_len:] = False
        # import pdb; pdb.set_trace()
        x[id_insert_mask] = face_token[id_insert_mask.any(dim=-1)]
       
        x = x + self.positional_embedding.to(cast_dtype)
        x = self.transformer(x, attn_mask=self.attn_mask) 
        x = self.ln_final(x)  # [batch_size, n_ctx, transformer.width]

        output_dict['text_embedding'] = x
        output_dict['attn_mask'] = self.attn_mask
        x, _ = text_global_pool(x, input_ids, self.text_pool_type)
        output_dict['pooled'] = x
        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                x = self.text_projection(x)
            else:
                x = x @ self.text_projection
        output_dict['pooled_after_projection'] = x
        output_dict['text_feature'] = F.normalize(x, dim=-1) if normalize else x
        return output_dict

    def create_text_self_attn(self, input_ids):
        context_length = self.tokenizer.context_length
        sot_token_id = self.tokenizer.sot_token_id
        eot_token_id = self.tokenizer.eot_token_id
        special_token_id = 0
        context_mask = (input_ids != sot_token_id) & (input_ids != eot_token_id) & (input_ids != special_token_id)
        context_mask = context_mask.unsqueeze(-1) * context_mask.unsqueeze(1)
        sot_mask = (input_ids == sot_token_id)
        sot_mask = sot_mask.unsqueeze(-1) * sot_mask.unsqueeze(1)
        eot_mask = (input_ids == eot_token_id)
        eot_mask = eot_mask.unsqueeze(-1) * eot_mask.unsqueeze(1)
        spe_mask = (input_ids == special_token_id)
        spe_mask = spe_mask.unsqueeze(-1) * spe_mask.unsqueeze(1)
        self_attn = context_mask | sot_mask | eot_mask
        return self_attn


    def forward_all(self, image, input_ids, face_token_id, id_text_mask, device='cpu'):
        batch_size = len(input_ids)
        if isinstance(image, list):
            image_ = torch.stack(image, dim=0)
        face_feature_ = self.face_encoder(
            F.interpolate(
                image_,
                size=(112, 112),
                mode='bilinear',
                align_corners=False
            ),
            dtype=image_.dtype
        )
        # face
        face_cls_embed, face_patch_embed = face_feature_['class_token'], face_feature_['patch_token']
        norm_face_cls_embed = face_cls_embed / face_cls_embed.norm(dim=-1, keepdim=True)
        proj_face_cls_embed = self.proj1(norm_face_cls_embed)
        proj_face_cls_embed = proj_face_cls_embed.to(face_cls_embed.dtype)
        
        face_pos_embedding = self.face_encoder.pos_embed.repeat(
            face_patch_embed.shape[0], 1, 1
        ).to(face_patch_embed.device)
        face_pos_embedding = self.face_pos_proj(face_pos_embedding)
        face_patch_embed = self.face_patch_proj(face_patch_embed)

        text_feature = self.encode_text(input_ids, id_text_mask, face_token_id, proj_face_cls_embed, normalize=True)
        text_embedding = self.text_embed_proj(text_feature['text_embedding'])
        text_pooled = text_feature['pooled']
        text_feat = text_feature['text_feature']
        text_pos_embedding = self.positional_embedding.unsqueeze(0).repeat(text_embedding.shape[0], 1, 1)
        text_pos_embedding = self.text_pos_proj(text_pos_embedding)
        text_self_attn_mask = self.create_text_self_attn(input_ids).to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            fused_feature = self.fuser(
                text_embedding,
                text_pos_embedding,
                text_self_attn_mask,
                norm_face_cls_embed,
                face_patch_embed,
                face_pos_embedding
            )
        return fused_feature, text_feat, norm_face_cls_embed, face_patch_embed, text_pooled

    def forward(self, aligned_face_image, face_exist_mask, text, return_feat=False, device='cpu'):
        # text
        id_symbol = 'id'
        id_text = "Photo of a id person."
        # insert id prompts
        text_ = [f"{id_text} " + t if m else t for t, m in zip(text, face_exist_mask)]
        face_token_id = self.tokenizer.encode(id_symbol)[0]
        input_ids = self.tokenizer(text_).to(device)

        # compute identity input mask
        id_input_ids = self.tokenizer(id_text).to(device)
        id_input_token_len = (id_input_ids != 0).sum(dim=-1)[0] - 1
        id_text_mask = torch.zeros([len(text_), self.context_length], dtype=torch.bool, device=device)
        id_text_mask[face_exist_mask, :id_input_token_len] = True

        fused_feature, text_feat, face_feat, face_patch_embed, text_pool = self.forward_all(aligned_face_image, input_ids, face_token_id, id_text_mask, device)

        if return_feat:
            return fused_feature, text_feat, face_feat, face_patch_embed
        else:
            return fused_feature, text_pool

def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class ImageBranch(nn.Module):
    def __init__(
            self,
            visual_embed_dim: int,
            quick_gelu: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            vision_model_name: str = '',
            pretrained: str = 'metaclip_fullcc'
    ):
        super().__init__()
        vision_model_config = get_model_config(vision_model_name)['vision_cfg']
        quick_gelu = get_model_config(vision_model_name).get('quick_gelu', False)
        self.vision = _build_vision_tower(visual_embed_dim, vision_model_config, quick_gelu, cast_dtype)
        self._init_parameters_from_preatrained(vision_model_name, pretrained)
        self._freeze_vision()

    def _init_parameters_from_preatrained(self, vision_model_name, pretrained):
        model, _, preprocess = open_clip.create_model_and_transforms(vision_model_name, pretrained=pretrained)
        self.preprocess = preprocess
        state_dict = model.visual.state_dict()
        # state_dict.pop('positional_embedding')
        missing, unexpected = self.vision.load_state_dict(state_dict, strict=True)
        print(f"Missing keys: {missing}")
        print(f"Unexpected keys: {unexpected}")

    def _freeze_vision(self):
        for param in self.vision.parameters():
            param.requires_grad = False

    def forward(self, image, normalize: bool = False):
        features = self.vision(image)
        return F.normalize(features, dim=-1) if normalize else features


class ImageTextEncoderWrapper(nn.Module):
    def __init__(self, ckpt_path, ckpt_spec):
        super().__init__()
        config = OmegaConf.load(os.path.join(ckpt_path, 'config.yaml'))
        self.model = instantiate_from_config(config.model.params.text_image_branch_config)

    def _init_model(self, ckpt):
        raise NotImplementedError



@dataclass
class TextModelOutput:
    embeddings: torch.Tensor
    masks: torch.Tensor
    pooled: List


class FaceCLIP_L_G_Wrapper(nn.Module):
    def __init__(self):
        super().__init__()

        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0',
            model_max_length=77,
            subfolder="tokenizer_2"
        )

        face_clip_l_config = OmegaConf.load('configs/face_clip_l_14_config.yaml')
        face_clip_g_config = OmegaConf.load('configs/face_clip_g_14_config.yaml')
        self.face_clip_l = instantiate_from_config(face_clip_l_config.model.text_image_branch_config)
        self.face_clip_g = instantiate_from_config(face_clip_g_config.model.text_image_branch_config)
        self._freeze_face_clip()

    def _freeze_face_clip(self):
        for param in self.face_clip_l.parameters():
            param.requires_grad = False
        for param in self.face_clip_g.parameters():
            param.requires_grad = False

    def forward(self, aligned_face_image, face_exist_mask, text, device=''):
        all_face_clip_embeds = []
        all_face_clip_pooleds = []
        with torch.no_grad():
            for face_clip_encoder in [self.face_clip_l, self.face_clip_g]:
                face_clip_embed, face_clip_pooled = face_clip_encoder(
                    aligned_face_image, face_exist_mask, text, return_feat=False, device=device
                )
                all_face_clip_embeds.append(face_clip_embed)
                all_face_clip_pooleds.append(face_clip_pooled)

        return TextModelOutput(
            embeddings=torch.cat(all_face_clip_embeds, dim=2),
            masks=None,
            pooled=[all_face_clip_pooleds[1]]
        )


