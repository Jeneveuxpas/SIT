# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# SiT with Encoder KV Distillation
# Two-stage training:
#   Stage 1: Q_SiT @ K_Enc, V_Enc (learn with Encoder semantics)
#   Stage 2: Q_SiT @ K_SiT, V_SiT + logits distillation loss
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict
from timm.models.vision_transformer import PatchEmbed, Mlp

from .sit import (
    TimestepEmbedder, LabelEmbedder, FinalLayer, 
    get_2d_sincos_pos_embed, modulate, ProjectionLayer,
    ALL_PROJECTION_LAYER_TYPES
)
from .encoder_adapter import EncoderKVProjection


class AttentionWithEncoderKV(nn.Module):
    """
    Attention module supporting two-stage Encoder-KV training.
    
    Stage 1: Q_sit @ K_enc^T -> Softmax -> V_enc
    Stage 2: Q_sit @ K_sit^T -> Softmax -> V_sit (with logits distillation loss)
    """
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        qkv_bias: bool = True,
        qk_norm: bool = False,
        fused_attn: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        
        # Optional QK norm
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor,
        k_enc: Optional[torch.Tensor] = None,
        v_enc: Optional[torch.Tensor] = None,
        stage: int = 2,
        align_mode: str = 'attn_mse',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward attention with staged Encoder-KV training.
        """
        B, N, C = x.shape
        
        # Compute Q, K, V from input
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k_sit, v_sit = qkv.unbind(0)
        
        # Apply QK norm
        q = self.q_norm(q)
        k_sit = self.k_norm(k_sit)
        
        distill_loss = None
        
        if stage == 1 and k_enc is not None and v_enc is not None:
            k = k_enc
            v = v_enc
        elif stage == 2 and k_enc is not None and v_enc is not None:
            # Stage 2: Use SiT K/V with alignment loss
            
            if align_mode == 'logits':
                logits_enc = (q @ k_enc.transpose(-2, -1)) * self.scale
                logits_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                logits_enc_n = F.normalize(logits_enc.float(), dim=-1)
                logits_sit_n = F.normalize(logits_sit.float(), dim=-1)
                distill_loss = 1 - F.cosine_similarity(logits_sit_n, logits_enc_n.detach(), dim=-1).mean()
                
            elif align_mode == 'attn_mse':
                if self.fused_attn:
                    # F.scaled_dot_product_attention might return different dtypes depending on impl, cast to float
                    attn_enc = F.scaled_dot_product_attention(q, k_enc, v_enc).float()
                    attn_sit = F.scaled_dot_product_attention(q, k_sit, v_sit).float()
                else:
                    attn_weights_enc = (q @ k_enc.transpose(-2, -1)) * self.scale
                    attn_weights_enc = attn_weights_enc.softmax(dim=-1)
                    attn_enc = (attn_weights_enc @ v_enc).float()
                    
                    attn_weights_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                    attn_weights_sit = attn_weights_sit.softmax(dim=-1)
                    attn_sit = (attn_weights_sit @ v_sit).float()
                
                distill_loss = F.mse_loss(attn_sit, attn_enc.detach())

            elif align_mode == 'kv_mse':
                k_loss = F.mse_loss(k_sit.float(), k_enc.float().detach())
                v_loss = F.mse_loss(v_sit.float(), v_enc.float().detach())
                distill_loss = k_loss + v_loss

            else:
                raise ValueError(f"Unknown align_mode: {align_mode}")
            
            k = k_sit
            v = v_sit
        else:
            k = k_sit
            v = v_sit
        
        # Attention
        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            x = attn @ v
        
        # Reshape and project
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x, distill_loss


class SiTBlockWithEncoderKV(nn.Module):
    """
    SiT block with optional Encoder K/V injection.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        enc_dim: Optional[int] = None,
        enc_heads: Optional[int] = None,
        kv_proj_type: str = "linear",
        kv_proj_hidden_dim: Optional[int] = None,
        kv_proj_kernel_size: int = 1,
        kv_norm_type: str = "zscore",
        kv_zscore_alpha: float = 1.0,
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithEncoderKV(
            hidden_size, 
            num_heads=num_heads, 
            qkv_bias=True, 
            qk_norm=block_kwargs.get("qk_norm", False),
            fused_attn=block_kwargs.get("fused_attn", True),
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        # Encoder K/V projection
        self.has_enc_kv = enc_dim is not None and enc_heads is not None
        if self.has_enc_kv:
            self.kv_proj = EncoderKVProjection(
                enc_dim, hidden_size, enc_heads, num_heads,
                kv_proj_type=kv_proj_type,
                kv_proj_hidden_dim=kv_proj_hidden_dim,
                kv_proj_kernel_size=kv_proj_kernel_size,
                kv_norm_type=kv_norm_type,
                kv_zscore_alpha=kv_zscore_alpha,
            )
        

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        enc_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stage: int = 2,
        align_mode: str = 'logits_attn',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        Args:
            enc_kv: Optional (K_enc, V_enc) tuple from encoder
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )     
        # Prepare Encoder K/V if available
        k_enc, v_enc = None, None
        if self.has_enc_kv and enc_kv is not None and self.training:
            k_raw, v_raw = enc_kv
            k_enc, v_enc = self.kv_proj(k_raw, v_raw, stage=stage)
        
        # Attention with stage-based K/V selection
        attn_out, distill_loss = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            k_enc=k_enc,
            v_enc=v_enc,
            stage=stage,
            align_mode=align_mode,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x, distill_loss


class SiTWithEncoderKV(nn.Module):
    """
    SiT with Encoder K/V Distillation.
    
    Structure:
    - Stage 1: Encoder K/V semantic guidance
    - Stage 2: SiT K/V + distillation
    """
    def __init__(
        self,
        path_type='edm',
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        decoder_hidden_size=768,
        encoder_depth=8,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        use_cfg=False,
        z_dims=[768],
        eval_mode=False,
        projector_dim=2048,
        projection_layer_type="mlp",
        proj_kwargs_kernel_size=3,
        # Encoder KV specific params
        enc_layer_indices: List[int] = [8],
        sit_layer_indices: List[int] = [8],
        enc_dim: int = 768,
        enc_heads: int = 12,
        # K/V projection config
        kv_proj_type: str = "linear",
        kv_proj_hidden_dim: Optional[int] = None,
        kv_proj_kernel_size: int = 3,
        kv_norm_type: str = "layernorm",
        kv_zscore_alpha: float = 1.0,
        **block_kwargs
    ):
        super().__init__()
        self.path_type = path_type
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_cfg = use_cfg
        self.num_classes = num_classes
        self.z_dims = z_dims
        self.encoder_depth = encoder_depth
        self.eval_mode = eval_mode
        self.projection_layer_type = projection_layer_type
        self.hidden_size = hidden_size
        
        # Encoder KV config
        self.enc_layer_indices = enc_layer_indices
        self.sit_layer_indices = sit_layer_indices
        self.enc_dim = enc_dim
        self.enc_heads = enc_heads
        
        # Mapping: SiT layer idx -> Encoder K/V list idx
        self.sit_to_enc_idx = {sit_idx: i for i, sit_idx in enumerate(sit_layer_indices)}

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in self.sit_to_enc_idx:
                block = SiTBlockWithEncoderKV(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    enc_dim=enc_dim, enc_heads=enc_heads,
                    kv_proj_type=kv_proj_type,
                    kv_proj_hidden_dim=kv_proj_hidden_dim,
                    kv_proj_kernel_size=kv_proj_kernel_size,
                    kv_norm_type=kv_norm_type,
                    kv_zscore_alpha=kv_zscore_alpha,
                    **block_kwargs
                )
            else:
                block = SiTBlockWithEncoderKV(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    enc_dim=None, enc_heads=None,
                    **block_kwargs
                )
            self.blocks.append(block)
        
        if not self.eval_mode:
            self.projectors = nn.ModuleList([
                ProjectionLayer(projection_layer_type, hidden_size=hidden_size, z_dim=z_dim, 
                                projector_dim=projector_dim, proj_kwargs_kernel_size=proj_kwargs_kernel_size) 
                for z_dim in z_dims
            ])
        
        self.final_layer = FinalLayer(decoder_hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x, patch_size=None):
        c = self.out_channels
        p = self.x_embedder.patch_size[0] if patch_size is None else patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: torch.Tensor, 
        y: torch.Tensor,
        enc_kv_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        stage: int = 2,
        align_mode: str = 'logits_attn',
        return_logvar: bool = False,
    ):
        """
        Forward pass.
        Args:
            enc_kv_list: List of (K, V) tuples from encoder extractor
        """
        x = self.x_embedder(x) + self.pos_embed
        N, T, D = x.shape
        
        t_embed = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t_embed + y
        
        zs = []
        zs_original = []
        accumulated_distill_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        
        # Track current index in enc_kv_list
        enc_list_idx = 0
        
        for i, block in enumerate(self.blocks):
            enc_kv = None
            if i in self.sit_to_enc_idx and enc_kv_list is not None:
                if enc_list_idx < len(enc_kv_list):
                    enc_kv = enc_kv_list[enc_list_idx]
                    enc_list_idx += 1
            
            x, block_distill_loss = block(
                x, c, enc_kv=enc_kv, stage=stage, align_mode=align_mode
            )
            
            if block_distill_loss is not None:
                accumulated_distill_loss = accumulated_distill_loss + block_distill_loss
            
            if (i + 1) == self.encoder_depth and not self.eval_mode:
                for projector in self.projectors:
                    if self.projection_layer_type in ["mlp", "linear"]:
                        z = projector(x.reshape(-1, D)).reshape(N, T, -1)
                    else:
                        z = projector(x)
                    zs.append(z)
                    zs_original.append(x.clone())
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        
        return x, zs, zs_original, accumulated_distill_loss


def SiT_XL_2_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiT_XL_4_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def SiT_XL_8_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def SiT_L_2_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiT_L_4_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def SiT_L_8_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def SiT_B_2_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiT_B_4_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def SiT_B_8_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def SiT_S_2_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def SiT_S_4_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def SiT_S_8_EncoderKV(**kwargs):
    return SiTWithEncoderKV(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


SiT_EncoderKV_models = {
    'SiT-XL/2-EncoderKV': SiT_XL_2_EncoderKV,  'SiT-XL/4-EncoderKV': SiT_XL_4_EncoderKV,  'SiT-XL/8-EncoderKV': SiT_XL_8_EncoderKV,
    'SiT-L/2-EncoderKV':  SiT_L_2_EncoderKV,   'SiT-L/4-EncoderKV':  SiT_L_4_EncoderKV,   'SiT-L/8-EncoderKV':  SiT_L_8_EncoderKV,
    'SiT-B/2-EncoderKV':  SiT_B_2_EncoderKV,   'SiT-B/4-EncoderKV':  SiT_B_4_EncoderKV,   'SiT-B/8-EncoderKV':  SiT_B_8_EncoderKV,
    'SiT-S/2-EncoderKV':  SiT_S_2_EncoderKV,   'SiT-S/4-EncoderKV':  SiT_S_4_EncoderKV,   'SiT-S/8-EncoderKV':  SiT_S_8_EncoderKV,
}
