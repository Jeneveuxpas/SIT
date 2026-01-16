# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# SiT with DINO-KV Distillation (extends iREPA's SiT)
# Two-stage training:
#   Stage 1: Q_SiT @ K_DINO, V_DINO (learn with DINO semantics)
#   Stage 2: Q_SiT @ K_SiT, V_SiT + logits distillation loss
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Tuple, Optional, Dict
from timm.models.vision_transformer import PatchEmbed, Mlp

from .sit import (
    TimestepEmbedder, LabelEmbedder, FinalLayer, 
    get_2d_sincos_pos_embed, modulate, ProjectionLayer,
    ALL_PROJECTION_LAYER_TYPES
)


class DinoKVExtractor(nn.Module):
    """
    Extract K/V from specified DINOv2 layers using forward hooks.
    
    DINOv2 uses `dinov2.layers.attention.Attention` which computes qkv internally.
    We hook into the attention forward to capture K/V before they're used.
    """
    def __init__(self, dino_model: nn.Module, layer_indices: List[int]):
        super().__init__()
        self.dino_model = dino_model
        self.layer_indices = layer_indices
        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._hooks = []
        
        # Register hooks
        self._register_hooks()
        
        # Freeze DINO
        for param in self.dino_model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks on specified DINO attention layers."""
        for idx in self.layer_indices:
            block = self.dino_model.blocks[idx]
            attn = block.attn
            
            # Create hook that captures K/V
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    # DINOv2 Attention: input is (x,) after norm
                    x = input[0]
                    B, N, C = x.shape
                    
                    # Recompute qkv to get K, V
                    qkv = module.qkv(x)
                    qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
                    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
                    q, k, v = qkv.unbind(0)
                    
                    # Remove CLS token (first token) - DINO has 257 tokens, we need 256
                    k = k[:, :, 1:, :]  # Skip CLS token
                    v = v[:, :, 1:, :]  # Skip CLS token
                    
                    # Store K, V (B, heads, num_patches, head_dim)
                    self.captured_kv[layer_idx] = (k.detach(), v.detach())
                return hook_fn
            
            hook = attn.register_forward_hook(make_hook(idx))
            self._hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through DINO to extract K/V from specified layers.
        
        Args:
            x: Input image tensor (B, C, H, W) preprocessed for DINO
            
        Returns:
            kv_list: List of (K, V) tuples for each specified layer
                K, V shape: (B, num_heads, num_patches, head_dim)
            cls_token: CLS token (B, dino_dim) for AdaLN modulation
        """
        self.captured_kv = {}
        
        # Forward through DINO and get CLS token
        output = self.dino_model.forward_features(x)
        cls_token = output['x_norm_clstoken']  # (B, dino_dim)
        
        # Collect K/V in order of layer_indices
        kv_list = []
        for idx in self.layer_indices:
            if idx in self.captured_kv:
                kv_list.append(self.captured_kv[idx])
            else:
                raise RuntimeError(f"K/V for layer {idx} not captured")
        
        return kv_list, cls_token



KV_PROJ_TYPES = ["linear", "mlp", "conv"]
KV_NORM_TYPES = ["none", "layernorm", "zscore", "batchnorm"]


def build_kv_norm(norm_type: str, dim: int, num_patches: int = 256):
    """
    Build normalization layer for K/V projection.
    
    Args:
        norm_type: "none", "layernorm", "zscore", "batchnorm"
        dim: Feature dimension
        num_patches: Number of patches (for BatchNorm1d)
    """
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "zscore":
        return ZScoreNorm()
    elif norm_type == "batchnorm":
        # BatchNorm over the feature dimension
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Unknown kv_norm_type: {norm_type}, must be one of {KV_NORM_TYPES}")


class ZScoreNorm(nn.Module):
    """Z-score normalization: (x - mean) / std per sample."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) or (B*N, D)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True) + self.eps
        return (x - mean) / std


def build_kv_mlp(in_dim: int, out_dim: int, hidden_dim: int = None):
    """Build MLP for K/V projection: in_dim -> hidden_dim -> hidden_dim -> out_dim"""
    if hidden_dim is None:
        hidden_dim = max(in_dim, out_dim)
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DinoKVProjection(nn.Module):
    """
    Project DINO K/V to SiT dimension.
    
    Supports multiple projection types:
    - "linear": Simple linear projection (default, backward compatible)
    - "mlp": Multi-layer perceptron with non-linearity
    - "conv": 2D convolution (preserves spatial locality)
    
    Supports multiple normalization types before projection:
    - "none": No normalization
    - "layernorm": Layer normalization (default for conv)
    - "zscore": Z-score normalization per sample
    - "batchnorm": Batch normalization
    
    Key feature: In Stage 2, the projection output is detached (no gradient).
    """
    def __init__(
        self, 
        dino_dim: int, 
        sit_dim: int, 
        dino_heads: int, 
        sit_heads: int,
        kv_proj_type: str = "linear",
        kv_proj_hidden_dim: int = None,
        kv_proj_kernel_size: int = 3,
        kv_norm_type: str = "layernorm",  # NEW: configurable normalization
    ):
        super().__init__()
        assert kv_proj_type in KV_PROJ_TYPES, f"kv_proj_type must be one of {KV_PROJ_TYPES}, got {kv_proj_type}"
        assert kv_norm_type in KV_NORM_TYPES, f"kv_norm_type must be one of {KV_NORM_TYPES}, got {kv_norm_type}"
        
        self.dino_dim = dino_dim
        self.sit_dim = sit_dim
        self.dino_heads = dino_heads
        self.sit_heads = sit_heads
        self.dino_head_dim = dino_dim // dino_heads
        self.sit_head_dim = sit_dim // sit_heads
        self.kv_proj_type = kv_proj_type
        self.kv_norm_type = kv_norm_type
        
        # Build projection layers based on type
        if kv_proj_type == "linear":
            self.proj_k = nn.Linear(dino_dim, sit_dim, bias=False)
            self.proj_v = nn.Linear(dino_dim, sit_dim, bias=False)
            # Initialize with small values
            nn.init.normal_(self.proj_k.weight, std=0.02)
            nn.init.normal_(self.proj_v.weight, std=0.02)
            
        elif kv_proj_type == "mlp":
            hidden_dim = kv_proj_hidden_dim or max(dino_dim, sit_dim)
            self.proj_k = build_kv_mlp(dino_dim, sit_dim, hidden_dim)
            self.proj_v = build_kv_mlp(dino_dim, sit_dim, hidden_dim)
            
        elif kv_proj_type == "conv":
            self.kv_proj_kernel_size = kv_proj_kernel_size
            padding = kv_proj_kernel_size // 2
            # Add configurable normalization before conv projection
            self.k_norm = build_kv_norm(kv_norm_type, dino_dim)
            self.v_norm = build_kv_norm(kv_norm_type, dino_dim)
            self.proj_k = nn.Conv2d(dino_dim, sit_dim, kernel_size=kv_proj_kernel_size, 
                                    stride=1, padding=padding, bias=False)
            self.proj_v = nn.Conv2d(dino_dim, sit_dim, kernel_size=kv_proj_kernel_size, 
                                    stride=1, padding=padding, bias=False)
    
    def _project_linear_or_mlp(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Project using linear or MLP: (B, N, D_in) -> (B, N, D_out)"""
        B, N, D = feat.shape
        out = proj(feat.reshape(B * N, D))
        return out.reshape(B, N, -1)
    
    def _project_conv(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Project using conv: (B, N, D_in) -> (B, N, D_out), with spatial reshape"""
        B, N, D = feat.shape
        H = W = int(N ** 0.5)
        assert H * W == N, f"Conv projection requires square grid, got N={N}"
        # (B, N, D) -> (B, D, H, W)
        feat_2d = feat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        out_2d = proj(feat_2d)  # (B, D_out, H, W)
        # (B, D_out, H, W) -> (B, N, D_out)
        return out_2d.permute(0, 2, 3, 1).reshape(B, N, -1)
    
    def forward(
        self, 
        k_dino: torch.Tensor, 
        v_dino: torch.Tensor,
        stage: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project DINO K/V to SiT dimension.
        
        Args:
            k_dino: (B, dino_heads, N, dino_head_dim)
            v_dino: (B, dino_heads, N, dino_head_dim)
            stage: Training stage. 1=trainable projection, 2=detached projection
            
        Returns:
            k_proj: (B, sit_heads, N, sit_head_dim)
            v_proj: (B, sit_heads, N, sit_head_dim)
        """
        B, _, N, _ = k_dino.shape
        
        # Reshape to (B, N, dino_dim) for projection
        k_flat = k_dino.transpose(1, 2).reshape(B, N, self.dino_dim)
        v_flat = v_dino.transpose(1, 2).reshape(B, N, self.dino_dim)
        
        # Project based on type
        if self.kv_proj_type in ("linear", "mlp"):
            k_proj = self._project_linear_or_mlp(k_flat, self.proj_k)
            v_proj = self._project_linear_or_mlp(v_flat, self.proj_v)
        elif self.kv_proj_type == "conv":
            # Apply LayerNorm before conv projection
            k_proj = self._project_conv(self.k_norm(k_flat), self.proj_k)
            v_proj = self._project_conv(self.v_norm(v_flat), self.proj_v)
        
        # Stage 2: Detach projection (no gradient through projection layer)
        if stage == 2:
            k_proj = k_proj.detach()
            v_proj = v_proj.detach()
        
        # Reshape to (B, sit_heads, N, sit_head_dim)
        k_proj = k_proj.reshape(B, N, self.sit_heads, self.sit_head_dim).transpose(1, 2)
        v_proj = v_proj.reshape(B, N, self.sit_heads, self.sit_head_dim).transpose(1, 2)
        
        return k_proj, v_proj




class AttentionWithDINOKV(nn.Module):
    """
    Attention module supporting two-stage DINO-KV training.
    
    Stage 1: Q_sit @ K_dino^T -> Softmax -> V_dino
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
        k_dino: Optional[torch.Tensor] = None,
        v_dino: Optional[torch.Tensor] = None,
        stage: int = 2,
        align_mode: str = 'logits_attn',
        kv_mode: str = 'kv',  # 'kv', 'k_only', 'v_only'
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward attention with staged DINO-KV training.
        
        Stage 1: Use DINO K/V directly (Q_SiT @ K_DINO, V_DINO)
        Stage 2: Use SiT K/V with alignment loss (mode-dependent)
        Inference: Pure SiT (no DINO)
        
        Args:
            x: Input tensor (B, N, C)
            k_dino: DINO K projected to SiT dim (B, heads, N, head_dim)
            v_dino: DINO V projected to SiT dim (B, heads, N, head_dim)
            stage: Training stage (1 or 2)
            align_mode: Alignment mode for Stage 2:
                - 'logits': Cosine similarity on Q@K^T logits only
                - 'logits_attn': Logits + normalized attention output (default)
                - 'attn_mse': MSE on attention outputs
                - 'kv_mse': Direct MSE on K and V
                - 'k_only': MSE on K only (V learns freely)
                - 'v_only': MSE on V only (K learns freely)
            kv_mode: Which components to replace/align:
                - 'kv': Both K and V
                - 'k_only': Only K
                - 'v_only': Only V
            
        Returns:
            output: Attention output (B, N, C)
            distill_loss: Distillation loss (Stage 2 only, else None)
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
        
        if stage == 1 and k_dino is not None and v_dino is not None:
            # Stage 1: Use DINO K/V directly for attention (based on kv_mode)
            if kv_mode == 'kv':
                k = k_dino
                v = v_dino
            elif kv_mode == 'k_only':
                k = k_dino
                v = v_sit  # 只替换 K，V 用 SiT 的
            elif kv_mode == 'v_only':
                k = k_sit  # 只替换 V，K 用 SiT 的
                v = v_dino
            else:
                k = k_dino
                v = v_dino
        elif stage == 2 and k_dino is not None and v_dino is not None:
            # Stage 2: Use SiT K/V with alignment loss
            
            if align_mode == 'logits':
                # Only logits alignment (weakest constraint)
                logits_dino = (q @ k_dino.transpose(-2, -1)) * self.scale
                logits_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                logits_dino_n = F.normalize(logits_dino, dim=-1)
                logits_sit_n = F.normalize(logits_sit, dim=-1)
                distill_loss = 1 - F.cosine_similarity(logits_sit_n, logits_dino_n.detach(), dim=-1).mean()
                
            elif align_mode == 'logits_attn':
                # Logits + normalized attention output (balanced)
                logits_dino = (q @ k_dino.transpose(-2, -1)) * self.scale
                logits_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                
                # Logits alignment
                logits_dino_n = F.normalize(logits_dino, dim=-1)
                logits_sit_n = F.normalize(logits_sit, dim=-1)
                logit_align_loss = 1 - F.cosine_similarity(logits_sit_n, logits_dino_n.detach(), dim=-1).mean()
                
                # Normalized attention output alignment
                attn_weights_dino = logits_dino.softmax(dim=-1)
                attn_weights_sit = logits_sit.softmax(dim=-1)
                attn_dino = attn_weights_dino @ v_dino
                attn_sit = attn_weights_sit @ v_sit
                
                attn_sit_n = F.normalize(attn_sit, dim=-1)
                attn_dino_n = F.normalize(attn_dino, dim=-1)
                attn_align_loss = 1 - F.cosine_similarity(attn_sit_n, attn_dino_n.detach(), dim=-1).mean()
                
                # Combined: logits (1.0) + normalized attn (0.5)
                distill_loss = 1.0 * logit_align_loss + 0.5 * attn_align_loss
                
            elif align_mode == 'attn_mse':
                # MSE on attention outputs (strong constraint)
                if self.fused_attn:
                    attn_dino = F.scaled_dot_product_attention(q, k_dino, v_dino)
                    attn_sit = F.scaled_dot_product_attention(q, k_sit, v_sit)
                else:
                    attn_weights_dino = (q @ k_dino.transpose(-2, -1)) * self.scale
                    attn_weights_dino = attn_weights_dino.softmax(dim=-1)
                    attn_dino = attn_weights_dino @ v_dino
                    
                    attn_weights_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                    attn_weights_sit = attn_weights_sit.softmax(dim=-1)
                    attn_sit = attn_weights_sit @ v_sit
                
                distill_loss = F.mse_loss(attn_sit, attn_dino.detach())

            elif align_mode == 'attn_cosine':
                # Cosine similarity on attention outputs (strong constraint)
                if self.fused_attn:
                    attn_dino = F.scaled_dot_product_attention(q, k_dino, v_dino)
                    attn_sit = F.scaled_dot_product_attention(q, k_sit, v_sit)
                else:
                    attn_weights_dino = (q @ k_dino.transpose(-2, -1)) * self.scale
                    attn_weights_dino = attn_weights_dino.softmax(dim=-1)
                    attn_dino = attn_weights_dino @ v_dino
                    
                    attn_weights_sit = (q @ k_sit.transpose(-2, -1)) * self.scale
                    attn_weights_sit = attn_weights_sit.softmax(dim=-1)
                    attn_sit = attn_weights_sit @ v_sit

                attn_sit_n = F.normalize(attn_sit, dim=-1)
                attn_dino_n = F.normalize(attn_dino, dim=-1)
                distill_loss = 1 - F.cosine_similarity(attn_sit_n, attn_dino_n.detach(), dim=-1).mean()

            elif align_mode == 'kv_mse':
                # Direct K/V MSE (strongest constraint)
                k_loss = F.mse_loss(k_sit, k_dino.detach())
                v_loss = F.mse_loss(v_sit, v_dino.detach())
                distill_loss = k_loss + v_loss
                
            elif align_mode == 'k_only':
                # Only K alignment (V learns freely)
                distill_loss = F.mse_loss(k_sit, k_dino.detach())
            
            elif align_mode == 'v_only':
                # Only V alignment (K learns freely)
                distill_loss = F.mse_loss(v_sit, v_dino.detach())
                
            else:
                raise ValueError(f"Unknown align_mode: {align_mode}")
            
            k = k_sit
            v = v_sit
        else:
            # Inference or no DINO: use SiT K/V
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


class SiTBlockWithDINOKV(nn.Module):
    """
    SiT block with optional DINO K/V injection.
    """
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int, 
        mlp_ratio: float = 4.0,
        dino_dim: Optional[int] = None,
        dino_heads: Optional[int] = None,
        kv_proj_type: str = "linear",
        kv_proj_hidden_dim: Optional[int] = None,
        kv_proj_kernel_size: int = 3,
        kv_norm_type: str = "layernorm",  # NEW: configurable normalization
        **block_kwargs
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = AttentionWithDINOKV(
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
        
        # DINO K/V projection (only if this block receives DINO K/V)
        self.has_dino_kv = dino_dim is not None and dino_heads is not None
        if self.has_dino_kv:
            self.kv_proj = DinoKVProjection(
                dino_dim, hidden_size, dino_heads, num_heads,
                kv_proj_type=kv_proj_type,
                kv_proj_hidden_dim=kv_proj_hidden_dim,
                kv_proj_kernel_size=kv_proj_kernel_size,
                kv_norm_type=kv_norm_type,
            )
        

    def forward(
        self, 
        x: torch.Tensor, 
        c: torch.Tensor,
        dino_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        stage: int = 2,
        align_mode: str = 'logits_attn',
        kv_mode: str = 'kv',
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C)
            c: Conditioning (timestep + class embed) (B, C)
            dino_kv: Optional (K_dino, V_dino) tuple from DINO
            stage: Training stage (1=use DINO K/V, 2=use SiT K/V with distillation)
            align_mode: Alignment mode for Stage 2
            kv_mode: 'kv', 'k_only', or 'v_only'
            
        Returns:
            x: Output tensor (B, N, C)
            distill_loss: Distillation loss (Stage 2 only)
        """
        # AdaLN modulation from timestep + class
        modulation = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation.chunk(6, dim=-1)
        
        # Prepare DINO K/V if available
        k_dino, v_dino = None, None
        if self.has_dino_kv and dino_kv is not None and self.training:
            k_raw, v_raw = dino_kv
            k_dino, v_dino = self.kv_proj(k_raw, v_raw, stage=stage)
        
        # Attention with stage-based K/V selection
        attn_out, distill_loss = self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa),
            k_dino=k_dino,
            v_dino=v_dino,
            stage=stage,
            align_mode=align_mode,
            kv_mode=kv_mode,
        )
        x = x + gate_msa.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x, distill_loss



class SiTWithDINOKV(nn.Module):
    """
    SiT with DINO-KV Distillation.
    
    Extends iREPA's SiT with two-stage DINO-KV training:
    - Stage 1: Use DINO K/V in attention (semantic guidance)
    - Stage 2: Use SiT K/V with logits distillation (structure alignment)
    
    REPA projection loss is preserved from original iREPA.
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
        # DINO-KV specific params
        dino_layer_indices: List[int] = [8],
        sit_layer_indices: List[int] = [8],
        dino_dim: int = 768,  # DINOv2-B dimension
        dino_heads: int = 12,  # DINOv2-B heads
        # K/V projection config
        kv_proj_type: str = "linear",  # "linear", "mlp", or "conv"
        kv_proj_hidden_dim: Optional[int] = None,  # MLP hidden dim
        kv_proj_kernel_size: int = 3,  # Conv kernel size
        kv_norm_type: str = "layernorm",  # NEW: "none", "layernorm", "zscore", "batchnorm"
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
        
        # DINO-KV config
        self.dino_layer_indices = dino_layer_indices
        self.sit_layer_indices = sit_layer_indices
        self.dino_dim = dino_dim
        self.dino_heads = dino_heads
        
        # Build mapping: SiT layer idx -> DINO K/V list idx
        self.sit_to_dino_idx = {sit_idx: i for i, sit_idx in enumerate(sit_layer_indices)}

        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_channels, hidden_size, bias=True
        )
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # Build blocks - some with DINO K/V support
        self.blocks = nn.ModuleList()
        for i in range(depth):
            if i in self.sit_to_dino_idx:
                # This block receives DINO K/V
                block = SiTBlockWithDINOKV(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    dino_dim=dino_dim, dino_heads=dino_heads,
                    kv_proj_type=kv_proj_type,
                    kv_proj_hidden_dim=kv_proj_hidden_dim,
                    kv_proj_kernel_size=kv_proj_kernel_size,
                    kv_norm_type=kv_norm_type,
                    **block_kwargs
                )
            else:
                # Standard block (no DINO K/V)
                block = SiTBlockWithDINOKV(
                    hidden_size, num_heads, mlp_ratio=mlp_ratio,
                    dino_dim=None, dino_heads=None,
                    **block_kwargs
                )
            self.blocks.append(block)
        
        # REPA projectors (from original iREPA)
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
            if isinstance(module, nn.Linear):
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
        dino_kv_list: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        stage: int = 2,
        align_mode: str = 'logits_attn',
        kv_mode: str = 'kv',
        return_logvar: bool = False,
    ):
        """
        Forward pass of SiT with staged DINO-KV training.
        
        Args:
            x: (B, C, H, W) latent images
            t: (B,) diffusion timesteps
            y: (B,) class labels
            dino_kv_list: List of (K, V) tuples from DINO extractor
            stage: Training stage (1=DINO K/V, 2=SiT K/V with distillation)
            align_mode: Alignment mode for Stage 2
                - 'logits': Cosine similarity on Q@K^T logits only
                - 'logits_attn': Logits + normalized attention output (default)
                - 'attn_mse': MSE on attention outputs
                - 'kv_mse': Direct MSE on K and V
                - 'k_only': MSE on K only
                - 'v_only': MSE on V only
            kv_mode: 'kv', 'k_only', or 'v_only'
            
        Returns:
            x: Output tensor (B, C, H, W)
            zs: Intermediate features for REPA projection loss
            zs_original: Original (unprojected) features  
            distill_loss: Accumulated distillation loss (Stage 2 only)
        """

        x = self.x_embedder(x) + self.pos_embed
        N, T, D = x.shape

        t_embed = self.t_embedder(t)
        y = self.y_embedder(y, self.training)
        c = t_embed + y

        # Accumulate distillation loss across layers
        total_distill_loss = 0.0
        num_distill_layers = 0

        zs = None
        zs_original = None

        for i, block in enumerate(self.blocks):
            # Get DINO K/V for this block if available
            dino_kv = None
            if dino_kv_list is not None and i in self.sit_to_dino_idx:
                dino_idx = self.sit_to_dino_idx[i]
                if dino_idx < len(dino_kv_list):
                    dino_kv = dino_kv_list[dino_idx]
            
            x, distill_loss = block(
                x, c, 
                dino_kv=dino_kv, 
                stage=stage, 
                align_mode=align_mode,
                kv_mode=kv_mode,
            )
            
            # Accumulate distillation loss
            if distill_loss is not None:
                total_distill_loss = total_distill_loss + distill_loss
                num_distill_layers += 1
            
            # REPA: Extract features at encoder_depth (from original iREPA)
            if (i + 1) == self.encoder_depth:
                if not self.eval_mode:
                    zs = [projector(x) for projector in self.projectors]
                    zs_original = [x.clone() for _ in self.projectors]
        
        x = self.final_layer(x, c)
        x = self.unpatchify(x)

        # Average the distillation loss across layers
        if num_distill_layers > 0:
            total_distill_loss = total_distill_loss / num_distill_layers

        return x, zs, zs_original, total_distill_loss


#################################################################################
#                              Model Configurations                             #
#################################################################################

def SiTWithDINOKV_XL_2(**kwargs):
    return SiTWithDINOKV(depth=28, hidden_size=1152, decoder_hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def SiTWithDINOKV_L_2(**kwargs):
    return SiTWithDINOKV(depth=24, hidden_size=1024, decoder_hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def SiTWithDINOKV_B_2(**kwargs):
    return SiTWithDINOKV(depth=12, hidden_size=768, decoder_hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def SiTWithDINOKV_S_2(**kwargs):
    return SiTWithDINOKV(depth=12, hidden_size=384, decoder_hidden_size=384, patch_size=2, num_heads=6, **kwargs)


SiT_DINOKV_models = {
    'SiT-XL/2-DINOKV': SiTWithDINOKV_XL_2,
    'SiT-L/2-DINOKV':  SiTWithDINOKV_L_2,
    'SiT-B/2-DINOKV':  SiTWithDINOKV_B_2,
    'SiT-S/2-DINOKV':  SiTWithDINOKV_S_2,
}