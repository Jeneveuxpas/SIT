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
    
    Supports:
    - Timm-style models (e.g. DINOv2 from torch.hub)
    - HF ViT/DINOv2 models (e.g. WebSSL)
    - SAM2 (Hiera) backbone
    
    Weights are frozen.
    """
    def __init__(self, encoder_model: nn.Module, layer_indices: List[int]):
        super().__init__()
        self.encoder_model = encoder_model
        self.layer_indices = layer_indices
        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.captured_feat: Dict[int, torch.Tensor] = {}
        self._hooks = []
        
        # Flatten blocks to allow index-based access
        self.blocks = self._get_model_blocks(encoder_model)
        
        # Register hooks
        self._register_hooks()
        
        # Freeze encoder
        for param in self.encoder_model.parameters():
            param.requires_grad = False
            
    def _get_model_blocks(self, model: nn.Module) -> List[nn.Module]:
        """Flatten model blocks into a list for consistent indexing."""
        # 1. Timm / TorchHub DINOv2
        if hasattr(model, "blocks"):
            return list(model.blocks)
        
        # 2. HF ViT / DINOv2 (WebSSL)
        # e.g. model.encoder.layer (ModuleList)
        if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            return list(model.encoder.layer)
        
        
        # 3. SAM2 (Hiera) - Option A: Direct blocks in backbone
        if hasattr(model, "backbone") and hasattr(model.backbone, "blocks"):
            return list(model.backbone.blocks)

        # 4. SAM2 (Hiera) - Option B: Stages
        # Structure: model.backbone.stages (ModuleList) -> each stage has blocks
        # We need to check if it's the specific SAM2 Vision Encoder structure
        if hasattr(model, "backbone") and hasattr(model.backbone, "stages"):
            blocks = []
            for stage in model.backbone.stages:
                # specific to Hiera implementation in transformers
                if hasattr(stage, "blocks"):
                    blocks.extend(list(stage.blocks))
                else: 
                     # Some implementations might behave differently, but Hiera usually has blocks
                     # Fallback or strict check
                     pass
            return blocks

        raise ValueError(f"Unsupported encoder architecture: {type(model)}")

    def _register_hooks(self):
        """Register forward hooks on specified encoder attention layers."""
        for idx in self.layer_indices:
            if idx >= len(self.blocks):
                raise ValueError(f"Layer index {idx} out of range (num_blocks={len(self.blocks)})")
                
            block = self.blocks[idx]
            
            # Identify attention module and Hook type
            # print(f"Inspecting block {idx} of type {type(block)}")
            
            # Case C: SAM2 Hiera -> has 'qkv' and 'query_stride' (specific to Hiera/SAM2)
            if hasattr(block, "attn") and hasattr(block.attn, "qkv") and hasattr(block.attn, "query_stride"):
                # print("Selected: SAM2 (qkv) hook")
                self._register_hf_sam2_qkv_hook(block.attn, idx)
            # Case A: Timm DINOv2 -> block.attn.qkv
            elif hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                # print("Selected: Timm hook")
                self._register_timm_hook(block.attn, idx)
            # Case B: HF DINOv2/ViT -> block.attention.attention.query/key/value
            elif hasattr(block, "attention") and hasattr(block.attention, "attention"):
                 # print("Selected: HF ViT hook")
                 self._register_hf_vit_hook(block.attention.attention, idx)
            # Case D: Generic HF SAM2 Hiera check (fallback for versions without qkv?)
            elif hasattr(block, "attn") and hasattr(block.attn, "q_proj"):
                # print("Selected: SAM2 (separate proj) hook")
                self._register_hf_sam2_hook(block.attn, idx)

            else:
                # Try to find something that looks like attention
                raise NotImplementedError(f"Could not find supported attention block in {type(block)}")
    
    def get_layer_dim(self, idx: int) -> int:
        """Get the embedding dimension of the specified layer."""
        if idx >= len(self.blocks):
            return 0
            
        block = self.blocks[idx]
        
        # SAM2 Hiera
        if hasattr(block, "attn") and hasattr(block.attn, "dim"):
            return block.attn.dim
        # Timm DINOv2
        elif hasattr(block, "attn") and hasattr(block.attn, "qkv"):
             if hasattr(block.attn, "dim"):
                 return block.attn.dim
             elif hasattr(block.attn, "qkv") and hasattr(block.attn.qkv, "in_features"):
                 return block.attn.qkv.in_features
        # HF ViT/DINOv2
        elif hasattr(block, "attention") and hasattr(block.attention, "attention"):
             # BERT/ViT style: attention.attention.key.in_features
             return block.attention.attention.key.in_features
             
        # Fallback: try to find linear layers in attention
        return 0

    def get_layer_heads(self, idx: int) -> int:
        """Get the number of attention heads of the specified layer."""
        if idx >= len(self.blocks):
            return 0
            
        block = self.blocks[idx]
        
        # SAM2 Hiera
        if hasattr(block, "attn") and hasattr(block.attn, "num_attention_heads"):
            return block.attn.num_attention_heads
        elif hasattr(block, "attn") and hasattr(block.attn, "num_heads"):
            return block.attn.num_heads
        # Timm DINOv2
        elif hasattr(block, "attn") and hasattr(block.attn, "num_heads"):
            return block.attn.num_heads
        # HF ViT/DINOv2
        elif hasattr(block, "attention") and hasattr(block.attention, "attention"):
            if hasattr(block.attention.attention, "num_attention_heads"):
                return block.attention.attention.num_attention_heads
            elif hasattr(block.attention.attention, "num_heads"):
                return block.attention.attention.num_heads
             
        # Fallback
        return 0
                
    def _register_timm_hook(self, attn_module, layer_idx):
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
            self.captured_feat[layer_idx] = x.detach()
        
        hook = attn_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _register_hf_vit_hook(self, attn_module, layer_idx):
        """Hook for HF ViT/DINOv2 (separated query/key/value layers)"""
        def hook_fn(module, input, output):
            # HF passes (hidden_states, ...)
            x = input[0]
            # In HF, internal attention module usually does the projection inside forward
            # But we are hooking the module that HAS .query, .key, .value
            # We can just manually call the projection layers
            
            # Get properties from module
            head_dim = module.head_dim if hasattr(module, 'head_dim') else (module.all_head_size // module.num_attention_heads)
            num_heads = module.num_attention_heads
            B, N, C = x.shape
            
            # Recompute K, V
            key_layer = module.key(x)
            value_layer = module.value(x)
            
            # Reshape: [B, N, heads, head_dim] -> transpose -> [B, heads, N, head_dim]
            k = key_layer.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = value_layer.view(B, N, num_heads, head_dim).transpose(1, 2)
            
            # Remove CLS token logic
            # HF ViT usually has CLS token at index 0
            # Warning: Some models might not. We assume standard ViT behavior here.
            # DINOv2 (WebSSL) has CLS token.
            k = k[:, :, 1:, :]
            v = v[:, :, 1:, :]
            
            self.captured_kv[layer_idx] = (k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        hook = attn_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _register_hf_sam2_hook(self, attn_module, layer_idx):
        """Hook for SAM2 Hiera Attention"""
        def hook_fn(module, args, kwargs, output):
            # Hiera forward(x, ...). x is [B, N, C]
            # Handle args or kwargs
            if len(args) > 0:
                x = args[0]
            elif 'hidden_states' in kwargs:
                x = kwargs['hidden_states']
            else:
                # Fallback or error
                # print("Warning: No input found in SAM2 hook")
                return

            B, N, C = x.shape
            
            # Check structure of HieraAttention
            # It has q_proj, k_proj, v_proj
            num_heads = module.num_heads
            head_dim = module.head_dim
            
            # Recompute K, V using the module's projections
            k = module.k_proj(x)
            v = module.v_proj(x)
            
            # Reshape [B, N, heads, head_dim] -> [B, heads, N, head_dim]
            k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N, num_heads, head_dim).transpose(1, 2)
            
            self.captured_kv[layer_idx] = (k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        # Use with_kwargs=True to capture named arguments (transformers often uses kwargs)
        hook = attn_module.register_forward_hook(hook_fn, with_kwargs=True)
        self._hooks.append(hook)

    def _register_hf_sam2_qkv_hook(self, attn_module, layer_idx):
        """Hook for SAM2 Hiera Attention with Fused QKV"""
        def hook_fn(module, args, kwargs, output):
            # Handle input
            if len(args) > 0:
                x = args[0]
            elif 'hidden_states' in kwargs:
                x = kwargs['hidden_states']
            else:
                return

            if len(x.shape) == 3:
                 B, N, C = x.shape
            elif len(x.shape) == 4:
                 # Handle 4D: (B, C, H, W) or (B, H, W, C)
                 # Determine channel dim
                 dim_val = module.dim if hasattr(module, 'dim') else (module.qkv.in_features if hasattr(module, 'qkv') else None)
                 
                 if dim_val and x.shape[1] == dim_val: # channels first (B, C, H, W)
                     x = x.flatten(2).transpose(1, 2) # -> B, N, C
                 elif dim_val and x.shape[3] == dim_val: # channels last (B, H, W, C)
                     x = x.flatten(1, 2) # -> B, N, C
                 elif dim_val is None:
                     # Fallback assumption: (B, C, H, W) is standard for many vision models, 
                     # but Hiera/SAM2 often uses (B, H, W, C).
                     # Based on debug (B, H, W, 384), it is channels last.
                     # Heuristic: smallest dim is usually channels? No, tokens are many.
                     # Heuristic: 3 for channels usually? No, embedding dim is large.
                     # Let's assume channels last if last dim > 3.
                     if x.shape[3] > 3:
                         x = x.flatten(1, 2)
                     else:
                         x = x.flatten(2).transpose(1, 2)
                 else:
                     # If dim match fails or ambiguous
                     if x.shape[3] == dim_val:
                         x = x.flatten(1, 2)
                     else:
                          x = x.flatten(2).transpose(1, 2)
                 
                 B, N, C = x.shape
            else:
                return


            
            # Recompute qkv
            # SAM2 Hiera qkv: Linear(dim, 3*dim, bias=True)
            # Output shape: [B, N, 3*dim]
            qkv = module.qkv(x)
            
            # Reshape to [B, N, 3, heads, head_dim]
            # Need to get num_heads. 
            num_heads = module.num_attention_heads
            head_dim = C // num_heads # Assuming dim_out == dim, usually true for attention blocks
            
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, heads, N, head_dim]
            q, k, v = qkv.unbind(0)
            
            self.captured_kv[layer_idx] = (k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        hook = attn_module.register_forward_hook(hook_fn, with_kwargs=True)
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
        self.captured_feat = {}
        
        # Forward through encoder and get CLS token
        if hasattr(self.encoder_model, "forward_features"):
            output = self.encoder_model.forward_features(x)
            # Timm DINOv2 returns dict if our wrapper, but raw timm model returns tensor or tuple
            # Wait, standard timm model returns tensor or (tensor, tensor).
            # But the 'vision_encoder.py' DINOEncoder.load_model uses torch.hub.
            # torch.hub DINOv2 has forward_features.
            if isinstance(output, dict):
                cls_token = output.get('x_norm_clstoken')
            else:
                # Assuming simple feature return if not dict
                cls_token = None
        else:
            # Fallback for HF models (forward)
            output = self.encoder_model(x)
            # HF models return output object or tuple
            # Attempt to extract CLS token if present (e.g. pooler_output)
            if hasattr(output, "pooler_output"):
                cls_token = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                # Check if it has CLS token at 0?
                # For ViT, yes. For SAM2, no.
                # We can try to infer from model type or just leave None
                cls_token = None
            else:
                cls_token = None

        
        
        # Collect K/V in order of layer_indices
        kv_list = []
        feat_list = []
        for idx in self.layer_indices:
            if idx in self.captured_kv:
                kv_list.append(self.captured_kv[idx])
                # Feature might be captured (if configured)
                if idx in self.captured_feat:
                    feat_list.append(self.captured_feat[idx])
                else:
                     # Fallback check
                     pass
            else:
                raise RuntimeError(f"K/V for layer {idx} not captured")
        
        return kv_list, cls_token


KV_PROJ_TYPES = ["linear", "mlp", "conv"]
KV_NORM_TYPES = ["none", "layernorm", "zscore", "zscore_token", "batchnorm"]


def build_kv_norm(norm_type: str, dim: int, num_patches: int = 256, alpha: float = 1.0):
    """
    Build normalization layer for K/V projection.
    """
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "zscore":
        return ZScoreNorm(dim=1, alpha=alpha)   # per-feature spatial normalization
    elif norm_type == "zscore_token":
        return ZScoreNorm(dim=-1, alpha=alpha)  # per-token normalization
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
