# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}  # (Q, K, V)
        self.captured_feat: Dict[int, torch.Tensor] = {}
        self.captured_attn_output: Dict[int, torch.Tensor] = {}
        self._hooks = []
        
        # Flatten blocks to allow index-based access
        self.blocks = self._get_model_blocks(encoder_model)
        
        # Register hooks
        self._register_hooks()
        
        # Freeze encoder
        for param in self.encoder_model.parameters():
            param.requires_grad = False

    def reset_cache(self):
        """Reset captured hook outputs before a new encoder forward."""
        self.captured_kv = {}
        self.captured_feat = {}
        self.captured_attn_output = {}

    def get_captured_kv_list(self) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Collect captured (Q, K, V) tuples in the same order as self.layer_indices.
        """
        kv_list = []
        for idx in self.layer_indices:
            if idx not in self.captured_kv:
                raise RuntimeError(f"Q/K/V for layer {idx} not captured")
            kv_list.append(self.captured_kv[idx])
        return kv_list

    def get_captured_feat_list(self) -> List[torch.Tensor]:
        """Collect patch-token features captured at the encoder attention inputs.

        Attention hooks keep the encoder's prefix tokens in ``captured_feat`` but
        remove them from Q/K/V.  Remove leading prefixes using model metadata (or
        square-grid inference), then spatially resize patch tokens when the
        selected encoder produces a different grid from the SiT block.

        Returns:
            List of tensors shaped ``(B, target_num_patches, encoder_dim)`` in
            the same order as ``self.layer_indices``.
        """
        feat_list = []
        for idx in self.layer_indices:
            if idx not in self.captured_feat:
                raise RuntimeError(f"Hidden features for layer {idx} not captured")
            feat_list.append(
                self._prepare_patch_features(
                    self.captured_feat[idx], idx, feature_name="hidden features"
                )
            )
        return feat_list

    def get_captured_attn_output_list(self) -> List[torch.Tensor]:
        """Collect encoder attention-branch outputs as spatial patch tokens.

        For timm-style DINOv2 this is the output of ``block.attn`` after its
        output projection and before the enclosing block applies LayerScale,
        residual addition, or the MLP.  Prefix-token outputs are removed and
        patch grids are resized using the same policy as attention-input
        features.
        """
        output_list = []
        for idx in self.layer_indices:
            if idx not in self.captured_attn_output:
                raise RuntimeError(
                    f"Attention output for layer {idx} not captured; "
                    "scaffold_feature_source=attn_output currently requires a "
                    "timm-style encoder attention hook"
                )
            output_list.append(
                self._prepare_patch_features(
                    self.captured_attn_output[idx],
                    idx,
                    feature_name="attention output",
                )
            )
        return output_list

    def _prepare_patch_features(
        self,
        feat: torch.Tensor,
        layer_idx: int,
        feature_name: str,
    ) -> torch.Tensor:
        """Remove prefix tokens and resize a captured encoder token grid."""
        if feat.ndim != 3:
            raise RuntimeError(
                f"Expected {feature_name} for layer {layer_idx} to have shape "
                f"(B, N, C), got {tuple(feat.shape)}"
            )

        original_dtype = feat.dtype
        kv_tokens = None
        if layer_idx in self.captured_kv:
            kv_tokens = int(self.captured_kv[layer_idx][1].shape[-2])

        num_prefix = 0
        metadata_prefix = self._get_prefix_token_count()
        if 0 < metadata_prefix < feat.shape[1]:
            remaining = feat.shape[1] - metadata_prefix
            remaining_hw = int(round(remaining ** 0.5))
            if remaining_hw * remaining_hw == remaining:
                num_prefix = metadata_prefix
        if num_prefix == 0:
            token_hw = int(round(feat.shape[1] ** 0.5))
            if token_hw * token_hw != feat.shape[1]:
                for candidate in range(1, min(17, feat.shape[1])):
                    patch_tokens = feat.shape[1] - candidate
                    patch_hw = int(round(patch_tokens ** 0.5))
                    if patch_hw * patch_hw == patch_tokens:
                        num_prefix = candidate
                        break
        if num_prefix > 0:
            if feat.shape[1] <= num_prefix:
                raise RuntimeError(
                    f"Invalid prefix-token count {num_prefix} for {feature_name} "
                    f"shape {tuple(feat.shape)} at layer {layer_idx}"
                )
            feat = feat[:, num_prefix:, :]

        configured_target = getattr(self, "_target_num_patches", None)
        target_tokens = configured_target or kv_tokens
        if target_tokens is not None and feat.shape[1] != target_tokens:
            source_hw = int(round(feat.shape[1] ** 0.5))
            target_hw = int(round(target_tokens ** 0.5))
            if source_hw * source_hw != feat.shape[1] or target_hw * target_hw != target_tokens:
                raise RuntimeError(
                    f"Cannot spatially resize encoder {feature_name} from "
                    f"{feat.shape[1]} to {target_tokens} non-square tokens"
                )
            feat_2d = feat.transpose(1, 2).reshape(
                feat.shape[0], feat.shape[2], source_hw, source_hw
            )
            feat_2d = F.interpolate(
                feat_2d.float(), size=(target_hw, target_hw),
                mode="bilinear", align_corners=False,
            )
            feat = feat_2d.flatten(2).transpose(1, 2).to(dtype=original_dtype)

        return feat.detach()
            
    def _get_model_blocks(self, model: nn.Module) -> List[nn.Module]:
        """Flatten model blocks into a list for consistent indexing."""
        # 0. CLIP UpdatedVisionTransformer wrapper
        if hasattr(model, "model") and hasattr(model.model, "transformer") and hasattr(model.model.transformer, "resblocks"):
            return list(model.model.transformer.resblocks)

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
            
            # Case E: CLIP -> block.attn is nn.MultiheadAttention with in_proj_weight
            if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
                self._register_clip_mha_hook(block.attn, idx)
            # Case C: SAM2 Hiera -> has 'qkv' AND is a Hiera/SAM2 block
            # Check query_stride (transition blocks) OR class name (all Hiera blocks)
            elif hasattr(block, "attn") and hasattr(block.attn, "qkv") and (
                hasattr(block.attn, "query_stride") or
                "hiera" in type(block).__name__.lower() or
                "sam2" in type(block).__name__.lower() or
                "hiera" in type(block.attn).__name__.lower() or
                "sam2" in type(block.attn).__name__.lower()
            ):
                self._register_hf_sam2_qkv_hook(block.attn, idx)
            # Case A: Timm DINOv2 -> block.attn.qkv
            elif hasattr(block, "attn") and hasattr(block.attn, "qkv"):
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

    def _get_prefix_token_count(self) -> int:
        prefix_tokens = getattr(self.encoder_model, "num_prefix_tokens", None)
        if prefix_tokens is not None:
            return int(prefix_tokens)
        cls_token = getattr(self.encoder_model, "cls_token", None)
        if cls_token is not None:
            return 1
        return 0
    
    def get_layer_dim(self, idx: int) -> int:
        """Get the embedding dimension of the specified layer's K/V output."""
        if idx >= len(self.blocks):
            return 0
            
        block = self.blocks[idx]
        
        # CLIP nn.MultiheadAttention
        if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
            return block.attn.embed_dim
        # SAM2 Hiera with fused QKV — use OUTPUT dim (qkv.out_features // 3)
        # because stage-transition blocks have qkv: Linear(dim_in, dim_out*3)
        elif hasattr(block, "attn") and hasattr(block.attn, "qkv") and hasattr(block.attn, "query_stride"):
            return block.attn.qkv.out_features // 3
        # Timm DINOv2
        elif hasattr(block, "attn") and hasattr(block.attn, "qkv"):
             if hasattr(block.attn, "dim"):
                 return block.attn.dim
             elif hasattr(block.attn.qkv, "in_features"):
                 return block.attn.qkv.in_features
        # HF ViT/DINOv2
        elif hasattr(block, "attention") and hasattr(block.attention, "attention"):
             # BERT/ViT style: attention.attention.key.in_features
             return block.attention.attention.key.in_features
        # SAM2 Hiera with separate projections
        elif hasattr(block, "attn") and hasattr(block.attn, "q_proj"):
            return block.attn.q_proj.out_features
             
        # Fallback: try to find linear layers in attention
        return 0

    def get_layer_input_dim(self, idx: int) -> int:
        """Get the channel dimension of the hidden state entering attention.

        This is usually identical to the Q/K/V output dimension, but can differ
        in hierarchical encoders at a stage-transition block.
        """
        if idx >= len(self.blocks):
            return 0

        block = self.blocks[idx]
        if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
            return block.attn.embed_dim
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            return int(block.attn.qkv.in_features)
        if hasattr(block, "attention") and hasattr(block.attention, "attention"):
            return int(block.attention.attention.query.in_features)
        if hasattr(block, "attn") and hasattr(block.attn, "q_proj"):
            return int(block.attn.q_proj.in_features)
        return self.get_layer_dim(idx)

    def get_layer_heads(self, idx: int) -> int:
        """Get the number of attention heads of the specified layer."""
        if idx >= len(self.blocks):
            return 0
            
        block = self.blocks[idx]
        
        # CLIP nn.MultiheadAttention
        if hasattr(block, "attn") and isinstance(block.attn, nn.MultiheadAttention):
            return block.attn.num_heads
        # SAM2 Hiera
        elif hasattr(block, "attn") and hasattr(block.attn, "num_attention_heads"):
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
                
    def _register_clip_mha_hook(self, attn_module, layer_idx):
        """Hook for CLIP nn.MultiheadAttention with fused in_proj_weight."""
        def hook_fn(module, input, output):
            # CLIP attention input: (query, key, value) all same tensor, shape (L, B, D)
            x = input[0]  # (L, B, D) — sequence-first format
            L, B, D = x.shape

            # Compute Q, K, V from in_proj_weight [3*D, D] and in_proj_bias [3*D]
            w = module.in_proj_weight
            b = module.in_proj_bias
            # Split into Q, K, V projections
            w_q, w_k, w_v = w[:D], w[D:2*D], w[2*D:3*D]
            b_q, b_k, b_v = b[:D], b[D:2*D], b[2*D:3*D]

            q = F.linear(x, w_q, b_q)  # (L, B, D)
            k = F.linear(x, w_k, b_k)  # (L, B, D)
            v = F.linear(x, w_v, b_v)  # (L, B, D)

            num_heads = module.num_heads
            head_dim = D // num_heads

            # Reshape: (L, B, D) -> (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
            q = q.permute(1, 0, 2).reshape(B, L, num_heads, head_dim).transpose(1, 2)
            k = k.permute(1, 0, 2).reshape(B, L, num_heads, head_dim).transpose(1, 2)
            v = v.permute(1, 0, 2).reshape(B, L, num_heads, head_dim).transpose(1, 2)

            # Remove CLS token (first token)
            q = q[:, :, 1:, :]
            k = k[:, :, 1:, :]
            v = v[:, :, 1:, :]

            self.captured_kv[layer_idx] = (q.detach(), k.detach(), v.detach())
            # Store feature in (B, L, D) format
            self.captured_feat[layer_idx] = x.permute(1, 0, 2).detach()

        hook = attn_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _register_timm_hook(self, attn_module, layer_idx):
        def hook_fn(module, input, output):
            # DINOv2 Attention: input is (x,) after norm
            x = input[0]
            B, N, C = x.shape
            
            # Recompute qkv to get Q, K, V
            qkv = module.qkv(x)
            qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
            q, k, v = qkv.unbind(0)
            
            # Remove prefix tokens if present (CLS/register tokens). Some timm
            # models (e.g. DeiT-III with ``no_embed_class``) keep a CLS token in
            # the attention sequence while their positional embedding is
            # patch-only, so ``num_prefix_tokens`` may incorrectly appear as 0
            # after positional-embedding adaptation. Fall back to square-grid
            # inference from the actual attention sequence.
            prefix_tokens = self._get_prefix_token_count()
            remaining = N - prefix_tokens
            remaining_hw = int(round(remaining ** 0.5))
            if remaining_hw * remaining_hw != remaining:
                for candidate in range(1, min(17, N)):
                    patch_tokens = N - candidate
                    patch_hw = int(round(patch_tokens ** 0.5))
                    if patch_hw * patch_hw == patch_tokens:
                        prefix_tokens = candidate
                        break
            if prefix_tokens > 0:
                q = q[:, :, prefix_tokens:, :]
                k = k[:, :, prefix_tokens:, :]
                v = v[:, :, prefix_tokens:, :]
            
            # Store Q, K, V (B, heads, num_patches, head_dim)
            self.captured_kv[layer_idx] = (q.detach(), k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

            # The timm/DINOv2 attention module returns its post-projection
            # branch output.  The enclosing block applies LayerScale and the
            # residual connection only after this hook fires.
            attn_output = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(attn_output) or attn_output.ndim != 3:
                raise RuntimeError(
                    f"Expected timm attention output with shape (B, N, C), got "
                    f"{type(attn_output)}"
                )
            self.captured_attn_output[layer_idx] = attn_output.detach()
        
        hook = attn_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _register_hf_vit_hook(self, attn_module, layer_idx):
        """Hook for HF ViT/DINOv2 (separated query/key/value layers)"""
        def hook_fn(module, input, output):
            # HF passes (hidden_states, ...)
            x = input[0]
            
            # Get properties from module
            head_dim = module.head_dim if hasattr(module, 'head_dim') else (module.all_head_size // module.num_attention_heads)
            num_heads = module.num_attention_heads
            B, N, C = x.shape
            
            # Recompute Q, K, V
            query_layer = module.query(x)
            key_layer = module.key(x)
            value_layer = module.value(x)
            
            # Reshape: [B, N, heads, head_dim] -> transpose -> [B, heads, N, head_dim]
            q = query_layer.view(B, N, num_heads, head_dim).transpose(1, 2)
            k = key_layer.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = value_layer.view(B, N, num_heads, head_dim).transpose(1, 2)
            
            # Remove CLS token (HF ViT / DINOv2 has CLS at index 0)
            q = q[:, :, 1:, :]
            k = k[:, :, 1:, :]
            v = v[:, :, 1:, :]
            
            self.captured_kv[layer_idx] = (q.detach(), k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        hook = attn_module.register_forward_hook(hook_fn)
        self._hooks.append(hook)

    def _register_hf_sam2_hook(self, attn_module, layer_idx):
        """Hook for SAM2 Hiera Attention"""
        def hook_fn(module, args, kwargs, output):
            # Hiera forward(x, ...). x is [B, N, C]
            if len(args) > 0:
                x = args[0]
            elif 'hidden_states' in kwargs:
                x = kwargs['hidden_states']
            else:
                return

            B, N, C = x.shape
            num_heads = module.num_heads
            head_dim = module.head_dim
            
            # Recompute Q, K, V using the module's projections
            q = module.q_proj(x)
            k = module.k_proj(x)
            v = module.v_proj(x)
            
            # Reshape [B, N, heads, head_dim] -> [B, heads, N, head_dim]
            q = q.view(B, N, num_heads, head_dim).transpose(1, 2)
            k = k.view(B, N, num_heads, head_dim).transpose(1, 2)
            v = v.view(B, N, num_heads, head_dim).transpose(1, 2)
            
            self.captured_kv[layer_idx] = (q.detach(), k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        hook = attn_module.register_forward_hook(hook_fn, with_kwargs=True)
        self._hooks.append(hook)

    def _register_hf_sam2_qkv_hook(self, attn_module, layer_idx):
        """Hook for SAM2 Hiera Attention with Fused QKV.
        
        Handles:
        - Windowed attention (B expanded by num_windows)
        - Stage-transition blocks (dim_in != dim_out in qkv linear)
        - Spatial interpolation to target token count
        """
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
                 dim_val = module.dim if hasattr(module, 'dim') else (module.qkv.in_features if hasattr(module, 'qkv') else None)
                 
                 if dim_val and x.shape[1] == dim_val:
                     x = x.flatten(2).transpose(1, 2)
                 elif dim_val and x.shape[3] == dim_val:
                     x = x.flatten(1, 2)
                 elif dim_val is None:
                     if x.shape[3] > 3:
                         x = x.flatten(1, 2)
                     else:
                         x = x.flatten(2).transpose(1, 2)
                 else:
                     if x.shape[3] == dim_val:
                         x = x.flatten(1, 2)
                     else:
                          x = x.flatten(2).transpose(1, 2)
                 
                 B, N, C = x.shape
            else:
                return

            # Recompute qkv
            qkv = module.qkv(x)
            
            # Get num_heads (attribute name varies across HF versions)
            if hasattr(module, 'num_attention_heads'):
                num_heads = module.num_attention_heads
            elif hasattr(module, 'num_heads'):
                num_heads = module.num_heads
            else:
                raise RuntimeError(f"Cannot find num_heads in {type(module)}")
            
            # Derive head_dim from QKV output, NOT input C.
            # Hiera stage-transition blocks have qkv: Linear(dim_in, dim_out*3)
            # where dim_out != dim_in (e.g. 384 -> 768*3=2304)
            dim_out = qkv.shape[-1] // 3
            head_dim = dim_out // num_heads
            
            qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4) # [3, B, heads, N, head_dim]
            q, k, v = qkv.unbind(0)
            
            # Un-window if windowed attention is used
            # (B will be B_orig * num_windows when windowed)
            B_orig = getattr(self, '_batch_size', None)
            if B_orig is not None and B > B_orig:
                q = self._unwindow_tensor(q, B_orig)
                k = self._unwindow_tensor(k, B_orig)
                v = self._unwindow_tensor(v, B_orig)
            
            # Spatially interpolate to target token count if needed
            target = getattr(self, '_target_num_patches', None)
            if target is not None and q.shape[2] != target:
                q = self._interpolate_tokens(q, target)
                k = self._interpolate_tokens(k, target)
                v = self._interpolate_tokens(v, target)
            
            self.captured_kv[layer_idx] = (q.detach(), k.detach(), v.detach())
            self.captured_feat[layer_idx] = x.detach()

        hook = attn_module.register_forward_hook(hook_fn, with_kwargs=True)
        self._hooks.append(hook)
    
    @staticmethod
    def _unwindow_tensor(tensor: torch.Tensor, B_orig: int) -> torch.Tensor:
        """
        Un-window: (B_orig*nH*nW, heads, ws*ws, head_dim) -> (B_orig, heads, Hp*Wp, head_dim)
        Reconstructs the full padded spatial grid from windowed attention output.
        """
        B_win, heads, N_win, head_dim = tensor.shape
        ws = int(N_win ** 0.5)
        
        num_windows = B_win // B_orig
        # Assume square window grid
        nH = nW = int(num_windows ** 0.5)
        if nH * nW != num_windows:
            # Non-square: try to factor
            for i in range(int(num_windows ** 0.5), 0, -1):
                if num_windows % i == 0:
                    nH, nW = i, num_windows // i
                    break
        
        # (B_orig*nH*nW, heads, ws, ws, head_dim)
        tensor = tensor.reshape(B_orig, nH, nW, heads, ws, ws, head_dim)
        # -> (B_orig, heads, nH, ws, nW, ws, head_dim)
        tensor = tensor.permute(0, 3, 1, 4, 2, 5, 6)
        # -> (B_orig, heads, Hp, Wp, head_dim)
        Hp, Wp = nH * ws, nW * ws
        tensor = tensor.reshape(B_orig, heads, Hp, Wp, head_dim)
        # -> (B_orig, heads, Hp*Wp, head_dim)
        tensor = tensor.reshape(B_orig, heads, Hp * Wp, head_dim)
        return tensor
    
    @staticmethod
    def _interpolate_tokens(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
        """
        Spatially interpolate tokens: (B, heads, N, head_dim) -> (B, heads, target_tokens, head_dim)
        Assumes square spatial layout.
        """
        B, heads, N, head_dim = tensor.shape
        H = W = int(N ** 0.5)
        tH = tW = int(target_tokens ** 0.5)
        
        if H * W != N or tH * tW != target_tokens:
            # Non-square, skip interpolation
            return tensor
        
        if H == tH and W == tW:
            return tensor
        
        # (B, heads, H, W, head_dim) -> (B*heads, head_dim, H, W)
        tensor = tensor.reshape(B, heads, H, W, head_dim)
        tensor = tensor.reshape(B * heads, H, W, head_dim).permute(0, 3, 1, 2)
        # Interpolate
        tensor = F.interpolate(tensor.float(), size=(tH, tW), mode='bilinear', align_corners=False)
        # -> (B, heads, tH, tW, head_dim) -> (B, heads, target_tokens, head_dim)
        tensor = tensor.permute(0, 2, 3, 1).reshape(B, heads, target_tokens, head_dim)
        return tensor

    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Tuple[List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]], torch.Tensor]:
        """
        Forward pass through encoder to extract Q/K/V from specified layers.
        
        Args:
            x: Input image tensor (B, C, H, W) preprocessed for encoder
            
        Returns:
            kv_list: List of (Q, K, V) tuples for each specified layer
                Q, K, V shape: (B, num_heads, num_patches, head_dim)
            cls_token: CLS token (B, enc_dim)
        """
        self.reset_cache()
        self._batch_size = x.shape[0]  # Store for un-windowing in hooks
        
        # Forward through encoder and get CLS token
        if hasattr(self.encoder_model, "forward_features"):
            output = self.encoder_model.forward_features(x)
            if isinstance(output, dict):
                cls_token = output.get('x_norm_clstoken')
            else:
                cls_token = None
        else:
            output = self.encoder_model(x)
            if hasattr(output, "pooler_output"):
                cls_token = output.pooler_output
            elif hasattr(output, "last_hidden_state"):
                cls_token = None
            else:
                cls_token = None
        
        # Collect Q/K/V in order of layer_indices
        kv_list = self.get_captured_kv_list()
        
        return kv_list, cls_token


KV_PROJ_TYPES = ["linear", "mlp", "conv", "conv_mlp3", "conv_mlp5", "head_gate"]
KV_NORM_TYPES = ["none", "layernorm", "rmsnorm", "zscore", "zscore_token", "batchnorm", "k_rms_v_layer"]


class TokenRMSNorm(nn.Module):
    """RMS-normalize each token without subtracting the feature mean."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms).to(dtype=dtype)


def build_kv_norm(norm_type: str, dim: int, num_patches: int = 256, alpha: float = 1.0):
    """
    Build normalization layer for K/V projection.
    """
    if norm_type == "none":
        return nn.Identity()
    elif norm_type == "layernorm":
        return nn.LayerNorm(dim)
    elif norm_type == "rmsnorm":
        return TokenRMSNorm()
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


class ConvResidualMLP5(nn.Module):
    """Spatial downsampling followed by a five-linear residual token MLP.

    The convolution performs the required 32x32 -> 16x16 grid conversion.  The
    five linear layers then learn the nonlinear VAE-feature -> attention-memory
    mapping independently for each projected component (K and V).
    """

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int):
        super().__init__()
        # Patch embedding uses non-overlapping k=stride windows.  Odd kernels
        # retain same-style padding for the existing overlapping projector.
        padding = 0 if kernel_size == stride else kernel_size // 2
        self.downsample = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.post_norm = nn.LayerNorm(out_dim)
        self.input_proj = nn.Linear(out_dim, out_dim)  # linear 1
        self.block1_norm = nn.LayerNorm(out_dim)
        self.block1_fc1 = nn.Linear(out_dim, out_dim)  # linear 2
        self.block1_fc2 = nn.Linear(out_dim, out_dim)  # linear 3
        self.block2_norm = nn.LayerNorm(out_dim)
        self.block2_fc1 = nn.Linear(out_dim, out_dim)  # linear 4
        self.block2_fc2 = nn.Linear(out_dim, out_dim)  # linear 5

        # Start each residual branch as identity while retaining a trainable
        # nonlinear input projection from the first update.
        nn.init.zeros_(self.block1_fc2.weight)
        nn.init.zeros_(self.block1_fc2.bias)
        nn.init.zeros_(self.block2_fc2.weight)
        nn.init.zeros_(self.block2_fc2.bias)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        x = self.downsample(feat_2d)
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = F.gelu(self.input_proj(self.post_norm(x)))
        residual = x
        x = self.block1_fc2(F.gelu(self.block1_fc1(self.block1_norm(x))))
        x = residual + x
        residual = x
        x = self.block2_fc2(F.gelu(self.block2_fc1(self.block2_norm(x))))
        x = residual + x
        return x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


class ConvResidualMLP3(nn.Module):
    """Spatial downsampling followed by a three-linear residual token MLP."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int, stride: int):
        super().__init__()
        padding = 0 if kernel_size == stride else kernel_size // 2
        self.downsample = nn.Conv2d(
            in_dim, out_dim, kernel_size=kernel_size,
            stride=stride, padding=padding, bias=False,
        )
        self.post_norm = nn.LayerNorm(out_dim)
        self.input_proj = nn.Linear(out_dim, out_dim)  # linear 1
        self.block_norm = nn.LayerNorm(out_dim)
        self.block_fc1 = nn.Linear(out_dim, out_dim)   # linear 2
        self.block_fc2 = nn.Linear(out_dim, out_dim)   # linear 3

        nn.init.zeros_(self.block_fc2.weight)
        nn.init.zeros_(self.block_fc2.bias)

    def forward(self, feat_2d: torch.Tensor) -> torch.Tensor:
        x = self.downsample(feat_2d)
        batch, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        x = F.gelu(self.input_proj(self.post_norm(x)))
        residual = x
        x = self.block_fc2(F.gelu(self.block_fc1(self.block_norm(x))))
        x = residual + x
        return x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()


KV_REPLACE_MODES = ["kv", "k", "v", "qkv", "qk", "q"]


class EncoderKVProjection(nn.Module):
    """
    Project Encoder Q/K/V to SiT dimension.
    
    Supports multiple projection types:
    - "linear": Simple linear projection (default)
    - "mlp": Multi-layer perceptron
    - "conv": 2D convolution
    - "conv_mlp3": Strided convolution plus three-linear residual token MLP
    - "conv_mlp5": Strided convolution plus five-linear residual token MLP
    
    Supports multiple normalization types before projection.
    
    Key feature: In Stage 2, the projection output is detached (no gradient).
    
    kv_replace_mode controls which components are projected:
    - "kv": Project K and V (default)
    - "k": Project only K
    - "v": Project only V
    - "qkv": Project Q, K, and V
    - "q": Project only Q
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
        kv_proj_stride: int = 1,
        kv_norm_type: str = "layernorm",
        kv_post_norm_type: str = "none",
        kv_zscore_alpha: float = 1.0,
        kv_replace_mode: str = "kv",
        kv_use_adaln: bool = False,
    ):
        super().__init__()
        assert kv_proj_type in KV_PROJ_TYPES, f"kv_proj_type must be one of {KV_PROJ_TYPES}, got {kv_proj_type}"
        assert kv_norm_type in KV_NORM_TYPES, f"kv_norm_type must be one of {KV_NORM_TYPES}, got {kv_norm_type}"
        assert kv_post_norm_type in KV_NORM_TYPES, f"kv_post_norm_type must be one of {KV_NORM_TYPES}, got {kv_post_norm_type}"
        assert kv_replace_mode in KV_REPLACE_MODES, f"kv_replace_mode must be one of {KV_REPLACE_MODES}, got {kv_replace_mode}"
        
        self.enc_dim = enc_dim
        self.sit_dim = sit_dim
        self.enc_heads = enc_heads
        self.sit_heads = sit_heads
        self.enc_head_dim = enc_dim // enc_heads
        self.sit_head_dim = sit_dim // sit_heads
        self.kv_proj_type = kv_proj_type
        self.kv_proj_stride = kv_proj_stride
        self.kv_norm_type = kv_norm_type
        self.kv_post_norm_type = kv_post_norm_type
        self.kv_replace_mode = kv_replace_mode
        self.kv_use_adaln = kv_use_adaln
        
        # Determine which components need projection
        self.need_q = kv_replace_mode in ("qkv", "qk", "q")
        self.need_k = kv_replace_mode in ("kv", "k", "qkv", "qk")
        self.need_v = kv_replace_mode in ("kv", "v", "qkv")

        def component_norm(component: str) -> nn.Module:
            norm_type = kv_norm_type
            if kv_norm_type == "k_rms_v_layer":
                norm_type = "layernorm" if component == "v" else "rmsnorm"
            return build_kv_norm(norm_type, enc_dim, alpha=kv_zscore_alpha)
        
        # Build normalization and projection layers for needed components
        if self.need_q:
            self.q_norm = component_norm("q")
        if self.need_k:
            self.k_norm = component_norm("k")
        if self.need_v:
            self.v_norm = component_norm("v")

        def post_component_norm(component: str) -> nn.Module:
            norm_type = kv_post_norm_type
            if kv_post_norm_type == "k_rms_v_layer":
                norm_type = "rmsnorm" if component == "k" else "layernorm"
            return build_kv_norm(norm_type, sit_dim, alpha=kv_zscore_alpha)

        if self.need_q:
            self.q_post_norm = post_component_norm("q")
        if self.need_k:
            self.k_post_norm = post_component_norm("k")
        if self.need_v:
            self.v_post_norm = post_component_norm("v")

        # Build projection layers based on type
        if kv_proj_type == "linear":
            if self.need_q:
                self.proj_q = nn.Linear(enc_dim, sit_dim, bias=False)
            if self.need_k:
                self.proj_k = nn.Linear(enc_dim, sit_dim, bias=False)
            if self.need_v:
                self.proj_v = nn.Linear(enc_dim, sit_dim, bias=False)

        elif kv_proj_type == "mlp":
            hidden_dim = kv_proj_hidden_dim or max(enc_dim, sit_dim)
            if self.need_q:
                self.proj_q = build_kv_mlp(enc_dim, sit_dim, hidden_dim)
            if self.need_k:
                self.proj_k = build_kv_mlp(enc_dim, sit_dim, hidden_dim)
            if self.need_v:
                self.proj_v = build_kv_mlp(enc_dim, sit_dim, hidden_dim)

        elif kv_proj_type == "conv":
            self.kv_proj_kernel_size = kv_proj_kernel_size
            padding = (
                0 if kv_proj_kernel_size == kv_proj_stride
                else kv_proj_kernel_size // 2
            )
            if self.need_q:
                self.proj_q = nn.Conv2d(enc_dim, sit_dim, kernel_size=kv_proj_kernel_size,
                                        stride=kv_proj_stride, padding=padding, bias=False)
            if self.need_k:
                self.proj_k = nn.Conv2d(enc_dim, sit_dim, kernel_size=kv_proj_kernel_size,
                                        stride=kv_proj_stride, padding=padding, bias=False)
            if self.need_v:
                self.proj_v = nn.Conv2d(enc_dim, sit_dim, kernel_size=kv_proj_kernel_size,
                                        stride=kv_proj_stride, padding=padding, bias=False)

        elif kv_proj_type in ("conv_mlp3", "conv_mlp5"):
            projector_cls = ConvResidualMLP3 if kv_proj_type == "conv_mlp3" else ConvResidualMLP5
            if self.need_q:
                self.proj_q = projector_cls(
                    enc_dim, sit_dim, kv_proj_kernel_size, kv_proj_stride
                )
            if self.need_k:
                self.proj_k = projector_cls(
                    enc_dim, sit_dim, kv_proj_kernel_size, kv_proj_stride
                )
            if self.need_v:
                self.proj_v = projector_cls(
                    enc_dim, sit_dim, kv_proj_kernel_size, kv_proj_stride
                )

        elif kv_proj_type == "head_gate":
            # Linear projection + t-conditioned per-head gating applied before projection.
            # gate = 1 + linear(silu(c)), zero-init -> all ones at start (identity).
            # Different t can emphasize different DINO heads (structural vs detail heads).
            if self.need_q:
                self.proj_q = nn.Linear(enc_dim, sit_dim, bias=False)
                self.head_gate_q = nn.Linear(sit_dim, enc_heads, bias=True)
                nn.init.zeros_(self.head_gate_q.weight)
                nn.init.zeros_(self.head_gate_q.bias)
            if self.need_k:
                self.proj_k = nn.Linear(enc_dim, sit_dim, bias=False)
                self.head_gate_k = nn.Linear(sit_dim, enc_heads, bias=True)
                nn.init.zeros_(self.head_gate_k.weight)
                nn.init.zeros_(self.head_gate_k.bias)
            if self.need_v:
                self.proj_v = nn.Linear(enc_dim, sit_dim, bias=False)
                self.head_gate_v = nn.Linear(sit_dim, enc_heads, bias=True)
                nn.init.zeros_(self.head_gate_v.weight)
                nn.init.zeros_(self.head_gate_v.bias)

        # FiLM-style t-conditioning: scale/shift on projected features, zero-init -> identity at start
        if kv_use_adaln:
            if self.need_q:
                self.adaLN_q = nn.Linear(sit_dim, 2 * sit_dim, bias=True)
                nn.init.zeros_(self.adaLN_q.weight)
                nn.init.zeros_(self.adaLN_q.bias)
            if self.need_k:
                self.adaLN_k = nn.Linear(sit_dim, 2 * sit_dim, bias=True)
                nn.init.zeros_(self.adaLN_k.weight)
                nn.init.zeros_(self.adaLN_k.bias)
            if self.need_v:
                self.adaLN_v = nn.Linear(sit_dim, 2 * sit_dim, bias=True)
                nn.init.zeros_(self.adaLN_v.weight)
                nn.init.zeros_(self.adaLN_v.bias)
    
    def _project_linear_or_mlp(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Project using linear or MLP: (B, N, D_in) -> (B, N, D_out)"""
        B, N, D = feat.shape
        out = proj(feat.reshape(B * N, D))
        return out.reshape(B, N, -1)
    
    def _project_conv(self, feat: torch.Tensor, proj: nn.Module) -> torch.Tensor:
        """Project a square token grid with conv, optionally changing its resolution."""
        B, N, D = feat.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"Conv KV projection requires square token count, got N={N}")
        # (B, N, D) -> (B, D, H, W)
        feat_2d = feat.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        out_2d = proj(feat_2d)
        # Strided projectors may change the token grid (e.g. VAE 32x32 -> SiT 16x16).
        out_h, out_w = out_2d.shape[-2:]
        return out_2d.permute(0, 2, 3, 1).reshape(B, out_h * out_w, -1)
    
    def _modulate(self, proj_out: torch.Tensor, adaLN: nn.Module, c: torch.Tensor) -> torch.Tensor:
        """FiLM modulation conditioned on t: zero-init -> identity at start.
        scale=0, shift=0 at init => output = proj_out (no distribution change)."""
        B, H, N, D = proj_out.shape
        flat = proj_out.transpose(1, 2).reshape(B, N, H * D)  # (B, N, sit_dim)
        scale, shift = adaLN(F.silu(c)).chunk(2, dim=-1)       # (B, sit_dim) each
        flat = flat * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return flat.reshape(B, N, H, D).transpose(1, 2)

    def _post_normalize(self, proj_out: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        """Normalize final projected memory over the complete SiT channel axis."""
        B, H, N, D = proj_out.shape
        flat = proj_out.transpose(1, 2).reshape(B, N, H * D)
        flat = norm(flat)
        return flat.reshape(B, N, H, D).transpose(1, 2)

    def _project_component_head_gate(
        self, enc_tensor: torch.Tensor, norm: nn.Module, proj: nn.Module,
        head_gate: nn.Module, c: torch.Tensor,
    ) -> torch.Tensor:
        """Project with t-conditioned per-head gating (before projection).
        gate = 1 + linear(silu(c)) per DINO head, zero-init -> identity at start.
        Allows different t to selectively emphasize structural vs detail DINO heads."""
        B, _, N, _ = enc_tensor.shape
        gate = 1.0 + head_gate(F.silu(c))          # (B, enc_heads)
        enc_gated = enc_tensor * gate[:, :, None, None]  # (B, enc_heads, N, enc_head_dim)
        flat = enc_gated.transpose(1, 2).reshape(B, N, self.enc_dim)  # (B, N, enc_dim)
        projected = self._project_linear_or_mlp(norm(flat), proj)     # (B, N, sit_dim)
        projected_tokens = projected.shape[1]
        return projected.reshape(
            B, projected_tokens, self.sit_heads, self.sit_head_dim
        ).transpose(1, 2)

    def _project_component(self, enc_tensor: torch.Tensor, norm: nn.Module, proj: nn.Module) -> torch.Tensor:
        """Project a single Q/K/V component: (B, enc_heads, N, enc_head_dim) -> (B, sit_heads, N, sit_head_dim)"""
        B, _, N, _ = enc_tensor.shape
        flat = enc_tensor.transpose(1, 2).reshape(B, N, self.enc_dim)
        
        if self.kv_proj_type in ("linear", "mlp"):
            projected = self._project_linear_or_mlp(norm(flat), proj)
        elif self.kv_proj_type in ("conv", "conv_mlp3", "conv_mlp5"):
            projected = self._project_conv(norm(flat), proj)
        
        projected_tokens = projected.shape[1]
        return projected.reshape(
            B, projected_tokens, self.sit_heads, self.sit_head_dim
        ).transpose(1, 2)
    
    def forward(
        self,
        q_enc: Optional[torch.Tensor] = None,
        k_enc: Optional[torch.Tensor] = None,
        v_enc: Optional[torch.Tensor] = None,
        stage: int = 1,
        c: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Project Encoder Q/K/V to SiT dimension based on kv_replace_mode.

        Args:
            q_enc: (B, enc_heads, N, enc_head_dim) - only used when mode includes Q
            k_enc: (B, enc_heads, N, enc_head_dim) - only used when mode includes K
            v_enc: (B, enc_heads, N, enc_head_dim) - only used when mode includes V
            stage: Training stage. 1=trainable projection, 2=detached projection
            c: (B, sit_dim) conditioning from t_embed + y_embed.
               Required for kv_proj_type="head_gate"; optional for kv_use_adaln.

        Returns:
            q_proj: (B, sit_heads, N, sit_head_dim) or None
            k_proj: (B, sit_heads, N, sit_head_dim) or None
            v_proj: (B, sit_heads, N, sit_head_dim) or None
        """
        q_proj, k_proj, v_proj = None, None, None

        if self.kv_proj_type == "head_gate":
            # t-conditioned per-head gating before projection; c is required
            if self.need_q and q_enc is not None:
                q_proj = self._project_component_head_gate(q_enc, self.q_norm, self.proj_q, self.head_gate_q, c)
            if self.need_k and k_enc is not None:
                k_proj = self._project_component_head_gate(k_enc, self.k_norm, self.proj_k, self.head_gate_k, c)
            if self.need_v and v_enc is not None:
                v_proj = self._project_component_head_gate(v_enc, self.v_norm, self.proj_v, self.head_gate_v, c)
        else:
            if self.need_q and q_enc is not None:
                q_proj = self._project_component(q_enc, self.q_norm, self.proj_q)
            if self.need_k and k_enc is not None:
                k_proj = self._project_component(k_enc, self.k_norm, self.proj_k)
            if self.need_v and v_enc is not None:
                v_proj = self._project_component(v_enc, self.v_norm, self.proj_v)

        # AdaLN modulation conditioned on t (via c = t_embed + y_embed)
        if self.kv_use_adaln and c is not None:
            if q_proj is not None:
                q_proj = self._modulate(q_proj, self.adaLN_q, c)
            if k_proj is not None:
                k_proj = self._modulate(k_proj, self.adaLN_k, c)
            if v_proj is not None:
                v_proj = self._modulate(v_proj, self.adaLN_v, c)

        # Final output normalization is intentionally after the projector and
        # optional AdaLN so the tensors entering attention have controlled scale.
        if q_proj is not None:
            q_proj = self._post_normalize(q_proj, self.q_post_norm)
        if k_proj is not None:
            k_proj = self._post_normalize(k_proj, self.k_post_norm)
        if v_proj is not None:
            v_proj = self._post_normalize(v_proj, self.v_post_norm)

        # Stage 2: Detach projection (no gradient through projection layer)
        if stage == 2:
            if q_proj is not None:
                q_proj = q_proj.detach()
            if k_proj is not None:
                k_proj = k_proj.detach()
            if v_proj is not None:
                v_proj = v_proj.detach()

        return q_proj, k_proj, v_proj

class LatentKVSource(nn.Module):
    """Clean VAE latent as scaffold source (DINO-free ablation).

    Patchify z_0 with the same patch size as SiT so token count matches exactly,
    then expose it in (B, enc_heads, N, enc_head_dim) layout so that the existing
    EncoderKVProjection can consume it unchanged.
    """
    def __init__(self, patch_size=2, in_channels=4, num_layers=1):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_layers = num_layers
        self.enc_dim = in_channels * patch_size * patch_size   # 4*2*2 = 16
        self.enc_heads = 1                                     # head_dim = enc_dim

    @torch.no_grad()
    def forward(self, z0):
        # z0: (B, C, H, W) clean latent, already scaled by sample_posterior
        B, C, H, W = z0.shape
        p = self.patch_size
        t = z0.reshape(B, C, H // p, p, W // p, p)
        t = t.permute(0, 2, 4, 1, 3, 5).reshape(B, (H // p) * (W // p), C * p * p)
        t = t.unsqueeze(1)                                     # (B, 1, N, 16)
        return [(t, t, t) for _ in range(self.num_layers)]




class VAEEncoderKVExtractor:
    """Capture Q/K/V from the VAE encoder's mid-block attention (AttnBlock).

    space-to-depth (patchify) by default: lossless, same operator as
    LatentKVSource, so the two ablations differ only in token content.
    """

    def __init__(self, vae_encoder, target_grid: int = 16, num_layers: int = 1,
                 token_mode: str = "patchify"):
        assert token_mode in ("patchify", "pool", "spatial")
        self.encoder = vae_encoder
        self.attn = vae_encoder.mid.attn_1
        self.target_grid = target_grid
        self.num_layers = num_layers
        self.token_mode = token_mode
        self._buf = {}
        self.hooks = [
            self.attn.q.register_forward_hook(self._make_hook("q")),
            self.attn.k.register_forward_hook(self._make_hook("k")),
            self.attn.v.register_forward_hook(self._make_hook("v")),
        ]

    def _make_hook(self, name):
        def hook(module, inputs, output):
            self._buf[name] = output
        return hook

    def _to_tokens(self, t):
        B, C, H, W = t.shape
        if self.token_mode == "spatial":
            return t.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
        if H == self.target_grid and W == self.target_grid:
            return t.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
        if self.token_mode == "pool":
            t = torch.nn.functional.adaptive_avg_pool2d(t.float(), self.target_grid).to(t.dtype)
            return t.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
        p = H // self.target_grid
        t = t.reshape(B, C, H // p, p, W // p, p).permute(0, 2, 4, 1, 3, 5)
        t = t.reshape(B, (H // p) * (W // p), C * p * p)
        return t.unsqueeze(1).contiguous()

    @torch.no_grad()
    def __call__(self, images):
        self._buf.clear()
        self.encoder(images)
        q = self._to_tokens(self._buf["q"]).detach()
        k = self._to_tokens(self._buf["k"]).detach()
        v = self._to_tokens(self._buf["v"]).detach()
        return [(q, k, v) for _ in range(self.num_layers)]

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


class VAEEncoderMidBlock2Extractor:
    """Expose the globally contextualized VAE ``mid.block_2`` output as memory."""

    def __init__(self, vae_encoder, num_layers: int = 1, norm_out_silu: bool = False):
        self.encoder = vae_encoder
        self.num_layers = num_layers
        self.norm_out_silu = norm_out_silu
        self._feature = None
        self.hook = vae_encoder.mid.block_2.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._feature = output

    @torch.no_grad()
    def __call__(self, images):
        self._feature = None
        self.encoder(images)
        if self._feature is None:
            raise RuntimeError("VAE mid.block_2 output was not captured")
        feature_2d = self._feature
        if self.norm_out_silu:
            # This matches the VAE encoder path immediately before conv_out.
            feature_2d = F.silu(self.encoder.norm_out(feature_2d))
        feature = feature_2d.flatten(2).transpose(1, 2).unsqueeze(1).contiguous()
        feature = feature.detach()  # (B, 1, 1024, 512)
        return [(None, feature, feature) for _ in range(self.num_layers)]

    def remove_hooks(self):
        self.hook.remove()
