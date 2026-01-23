# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
from utils import ZScoreNorm


class EncoderKVExtractor(nn.Module):
    """
    Extract K/V from specified encoder layers using forward hooks.
    
    Currently assumes DINOv2 structure but named generically.
    Weights are frozen.
    """
    def __init__(self, encoder_model: nn.Module, layer_indices: List[int]):
        super().__init__()
        self.encoder_model = encoder_model
        self.layer_indices = layer_indices
        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._hooks = []
        
        # Register hooks
        self._register_hooks()
        
        # Freeze encoder
        for param in self.encoder_model.parameters():
            param.requires_grad = False
    
    def _register_hooks(self):
        """Register forward hooks on specified encoder attention layers."""
        for idx in self.layer_indices:
            # Assuming DINOv2 structure for now: blocks[i].attn.qkv
            # TODO: Abstraction for other encoders
            block = self.encoder_model.blocks[idx]
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
        Forward pass through encoder to extract K/V from specified layers.
        
        Args:
            x: Input image tensor (B, C, H, W) preprocessed for encoder
            
        Returns:
            kv_list: List of (K, V) tuples for each specified layer
                K, V shape: (B, num_heads, num_patches, head_dim)
            cls_token: CLS token (B, enc_dim)
        """
        self.captured_kv = {}
        
        # Forward through encoder and get CLS token
        output = self.encoder_model.forward_features(x)
        cls_token = output['x_norm_clstoken']  # (B, enc_dim)
        
        # Collect K/V in order of layer_indices
        kv_list = []
        for idx in self.layer_indices:
            if idx in self.captured_kv:
                kv_list.append(self.captured_kv[idx])
            else:
                raise RuntimeError(f"K/V for layer {idx} not captured")
        
        return kv_list, cls_token


KV_PROJ_TYPES = ["linear", "mlp", "conv"]
KV_NORM_TYPES = ["none", "layernorm", "zscore", "zscore_spatial", "batchnorm"]


def build_kv_norm(norm_type: str, dim: int, num_patches: int = 256, alpha: float = 1.0):
    """
    Build normalization layer for K/V projection.
    """
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "zscore":
        return ZScoreNorm(dim=-1, alpha=alpha)  # per-token normalization
    elif norm_type == "zscore_spatial":
        return ZScoreNorm(dim=1, alpha=alpha)   # per-feature spatial normalization
    elif norm_type == "batchnorm":
        return nn.BatchNorm1d(dim)
    else:
        raise ValueError(f"Unknown kv_norm_type: {norm_type}, must be one of {KV_NORM_TYPES}")


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


class EncoderKVProjection(nn.Module):
    """
    Project Encoder K/V to SiT dimension.
    
    Supports multiple projection types:
    - "linear": Simple linear projection (default)
    - "mlp": Multi-layer perceptron
    - "conv": 2D convolution
    
    Supports multiple normalization types before projection.
    
    Key feature: In Stage 2, the projection output is detached (no gradient).
    """
    def __init__(
        self, 
        enc_dim: int, 
        sit_dim: int, 
        enc_heads: int, 
        sit_heads: int,
        kv_proj_type: str = "linear",
        kv_proj_hidden_dim: int = None,
        kv_proj_kernel_size: int = 3,
        kv_norm_type: str = "layernorm",
        kv_zscore_alpha: float = 1.0,
    ):
        super().__init__()
        assert kv_proj_type in KV_PROJ_TYPES, f"kv_proj_type must be one of {KV_PROJ_TYPES}, got {kv_proj_type}"
        assert kv_norm_type in KV_NORM_TYPES, f"kv_norm_type must be one of {KV_NORM_TYPES}, got {kv_norm_type}"
        
        self.enc_dim = enc_dim
        self.sit_dim = sit_dim
        self.enc_heads = enc_heads
        self.sit_heads = sit_heads
        self.enc_head_dim = enc_dim // enc_heads
        self.sit_head_dim = sit_dim // sit_heads
        self.kv_proj_type = kv_proj_type
        self.kv_norm_type = kv_norm_type
        
        # Common normalization layers
        self.k_norm = build_kv_norm(kv_norm_type, enc_dim, alpha=kv_zscore_alpha)
        self.v_norm = build_kv_norm(kv_norm_type, enc_dim, alpha=kv_zscore_alpha)

        # Build projection layers based on type
        if kv_proj_type == "linear":
            self.proj_k = nn.Linear(enc_dim, sit_dim, bias=False)
            self.proj_v = nn.Linear(enc_dim, sit_dim, bias=False)
            nn.init.normal_(self.proj_k.weight, std=0.02)
            nn.init.normal_(self.proj_v.weight, std=0.02)
            
        elif kv_proj_type == "mlp":
            hidden_dim = kv_proj_hidden_dim or max(enc_dim, sit_dim)
            self.proj_k = build_kv_mlp(enc_dim, sit_dim, hidden_dim)
            self.proj_v = build_kv_mlp(enc_dim, sit_dim, hidden_dim)
            
        elif kv_proj_type == "conv":
            self.kv_proj_kernel_size = kv_proj_kernel_size
            padding = kv_proj_kernel_size // 2
            # Norms are already initialized above
            self.proj_k = nn.Conv2d(enc_dim, sit_dim, kernel_size=kv_proj_kernel_size, 
                                    stride=1, padding=padding, bias=False)
            self.proj_v = nn.Conv2d(enc_dim, sit_dim, kernel_size=kv_proj_kernel_size, 
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
        # (B, N, D) -> (B, D, H, W)
        feat_2d = feat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        out_2d = proj(feat_2d)  # (B, D_out, H, W)
        # (B, D_out, H, W) -> (B, N, D_out)
        return out_2d.permute(0, 2, 3, 1).reshape(B, N, -1)
    
    def forward(
        self, 
        k_enc: torch.Tensor, 
        v_enc: torch.Tensor,
        stage: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project Encoder K/V to SiT dimension.
        
        Args:
            k_enc: (B, enc_heads, N, enc_head_dim)
            v_enc: (B, enc_heads, N, enc_head_dim)
            stage: Training stage. 1=trainable projection, 2=detached projection
            
        Returns:
            k_proj: (B, sit_heads, N, sit_head_dim)
            v_proj: (B, sit_heads, N, sit_head_dim)
        """
        B, _, N, _ = k_enc.shape
        
        # Reshape to (B, N, enc_dim) for projection
        k_flat = k_enc.transpose(1, 2).reshape(B, N, self.enc_dim)
        v_flat = v_enc.transpose(1, 2).reshape(B, N, self.enc_dim)
        
        # Project based on type
        if self.kv_proj_type in ("linear", "mlp"):
            k_proj = self._project_linear_or_mlp(self.k_norm(k_flat), self.proj_k)
            v_proj = self._project_linear_or_mlp(self.v_norm(v_flat), self.proj_v)
        elif self.kv_proj_type == "conv":
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
