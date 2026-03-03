#!/usr/bin/env python3
"""
Compute Spatial Structure Metrics (LDS, CDS, RMSC) on DiT/SiT Hidden States.

This script evaluates how well SiT's internal representations preserve spatial structure
by computing metrics on activations from intermediate layers at various timesteps.

Usage:
    CUDA_VISIBLE_DEVICES=7 python scripts/compute_spatial_metrics.py \
        --checkpoint /workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt \
        --data-dir /dev/shm/data \
        --num-samples 256 \
        --device cuda
"""

import os
import sys
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.sit import SiT_models
from models.sit_encoder import SiT_EncoderKV_models
from models.autoencoder import VAE_F8D4
from dataset import HFImgLatentDataset, ImageFolderLatentDataset


#################################################################################
#                         Spatial Metrics Implementation                        #
#################################################################################

class _GridCache:
    """Cache for grid coordinates and Manhattan distances."""
    
    def __init__(self):
        self._dist = {}   # key: (H,W,device) -> (T,T) manhattan dist
        self._coords = {} # key: (H,W,device) -> (T,2)

    def coords(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._coords:
            yy, xx = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            self._coords[key] = torch.stack([yy.flatten(), xx.flatten()], dim=1)  # (T,2)
        return self._coords[key]

    def dist(self, H: int, W: int, device: torch.device):
        key = (H, W, str(device))
        if key not in self._dist:
            c = self.coords(H, W, device)  # (T,2)
            self._dist[key] = (c[:, None, :] - c[None, :, :]).abs().sum(-1)  # (T,T) manhattan
        return self._dist[key]


_GRID = _GridCache()


def _normalize(u: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """L2 normalize along last dimension."""
    return F.normalize(u.to(torch.float32), dim=-1, eps=eps)


def _reshape_hw(u: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """Reshape (B,T,D) -> (B,H,W,D)."""
    return u.view(u.size(0), H, W, u.size(-1))


def _gram_cos(u: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity Gram matrix. u: (B,T,D) already normalized -> (B,T,T)."""
    return torch.einsum("btd,bsd->bts", u, u)


@torch.no_grad()
def metric_lds(u: torch.Tensor, H: int, W: int, far_dist: int = 6) -> torch.Tensor:
    """
    Local-vs-Distant Similarity (LDS):
    Mean cosine over 4-neighbors minus mean cosine over pairs with Manhattan distance >= far_dist.
    
    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        far_dist: Minimum Manhattan distance for "far" pairs
        
    Returns:
        (B,) tensor of LDS scores per sample
    """
    B, T, D = u.shape
    u = _normalize(u)
    uHW = _reshape_hw(u, H, W)  # (B,H,W,D)

    # local (4-neighbors) via shifts
    up = (uHW[:, 1:, :, :] * uHW[:, :-1, :, :]).sum(-1)    # (B,H-1,W)
    left = (uHW[:, :, 1:, :] * uHW[:, :, :-1, :]).sum(-1)  # (B,H,W-1)
    local = torch.cat([up.flatten(1), left.flatten(1)], dim=1).mean(dim=1)  # (B,)

    # far
    G = _gram_cos(u)  # (B,T,T)
    dist = _GRID.dist(H, W, u.device)
    far_mask = (dist >= far_dist) & (dist > 0)
    denom = far_mask.sum().clamp(min=1)
    far = (G.masked_fill(~far_mask, 0.0).sum(dim=(1, 2)) / denom)  # (B,)

    return local - far


@torch.no_grad()
def metric_cds(u: torch.Tensor, H: int, W: int, dmax: int = 8) -> torch.Tensor:
    """
    Correlation-Decay Slope (CDS):
    Fit a line to mean cosine vs Manhattan distance d=1..dmax; return -slope.
    
    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        dmax: Maximum distance for fitting
        
    Returns:
        (B,) tensor of CDS scores per sample
    """
    B, T, D = u.shape
    u = _normalize(u)
    G = _gram_cos(u)  # (B,T,T)
    dist = _GRID.dist(H, W, u.device)

    sims = []
    ds = []
    for d in range(1, dmax + 1):
        m = (dist == d)
        if m.any():
            sims.append(G[:, m].mean(dim=1))  # (B,)
            ds.append(d)
    if len(sims) == 0:
        return torch.zeros(B, device=u.device)

    S = torch.stack(sims, dim=1)  # (B,K)
    x = torch.tensor(ds, device=u.device, dtype=S.dtype)  # (K,)
    x0 = x - x.mean()
    denom = (x0 @ x0).clamp(min=1e-12)
    # slope per sample
    b = ((S - S.mean(dim=1, keepdim=True)) @ x0) / denom  # (B,)
    return -b


@torch.no_grad()
def metric_rmsc(u: torch.Tensor, H: int, W: int, sqrt: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """
    RMS Spatial Contrast (RMSC):
    Measures spread/diversity of token representations.
    
    Args:
        u: Features tensor of shape (B, T, D)
        H, W: Height and width of spatial grid
        sqrt: Whether to take sqrt of mean squared norm
        eps: Small epsilon for numerical stability
        
    Returns:
        (B,) tensor of RMSC scores per sample
    """
    x = F.normalize(u.to(torch.float32), dim=-1, eps=eps)  # L2 over D (per token)
    xc = x - x.mean(dim=1, keepdim=True)                   # mean-center over tokens
    ms = xc.square().sum(dim=-1).mean(dim=1)               # mean ||.||^2 across tokens
    return ms.sqrt() if sqrt else ms                        # (B,)


def compute_spatial_metrics(
    hidden_states: torch.Tensor,
    H: int,
    W: int,
) -> Dict[str, torch.Tensor]:
    """
    Compute all spatial metrics for hidden states.
    
    Args:
        hidden_states: (B, T, D) tensor of hidden states
        H, W: Grid dimensions
        
    Returns:
        Dict with keys 'lds', 'cds', 'rmsc', each containing per-sample values
    """
    return {
        'lds': metric_lds(hidden_states, H, W, far_dist=6),
        'cds': metric_cds(hidden_states, H, W, dmax=8),
        'rmsc': metric_rmsc(hidden_states, H, W),
    }


#################################################################################
#                              Evaluation Functions                             #
#################################################################################

@torch.no_grad()
def sample_posterior(moments: torch.Tensor, latents_scale: float = 1., latents_bias: float = 0.):
    """Sample from VAE posterior."""
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: str = "cuda",
) -> Tuple[torch.nn.Module, dict]:
    """Load model from checkpoint."""
    print(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    args = ckpt.get('args', None)
    if args is None:
        raise ValueError("Checkpoint does not contain 'args'. Cannot determine model configuration.")
    
    # Determine model type
    model_name = args.model if hasattr(args, 'model') else 'SiT-B/2'
    
    # Check if it's EncoderKV model
    if 'EncoderKV' in model_name:
        model_fn = SiT_EncoderKV_models.get(model_name)
    else:
        model_fn = SiT_models.get(model_name)
    
    if model_fn is None:
        # Fallback to base model name
        base_name = model_name.replace('-EncoderKV', '')
        model_fn = SiT_models.get(base_name)
    
    # Build model with eval_mode=True to skip projectors
    block_kwargs = {
        "fused_attn": getattr(args, 'fused_attn', True),
        "qk_norm": getattr(args, 'qk_norm', False),
    }
    
    model = model_fn(
        input_size=32,  # 256 // 8
        in_channels=4,
        num_classes=getattr(args, 'num_classes', 1000),
        use_cfg=getattr(args, 'cfg_prob', 0) > 0,
        z_dims=[768],
        encoder_depth=getattr(args, 'encoder_depth', 8),
        eval_mode=True,
        **block_kwargs
    )
    
    # Load state dict
    state_dict = ckpt.get('ema', ckpt.get('model'))
    
    # Filter out projector weights since we're in eval_mode
    filtered_state_dict = {k: v for k, v in state_dict.items() if 'projector' not in k}
    
    model.load_state_dict(filtered_state_dict, strict=False)
    model = model.to(device).eval()
    
    return model, args


def extract_hidden_states(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    layer_depths: List[int],
) -> List[torch.Tensor]:
    """
    Extract hidden states at specified layer depths.
    
    Args:
        model: SiT model
        x: Noisy latent (B, C, H, W)
        t: Timesteps (B,)
        y: Class labels (B,)
        layer_depths: List of layer depths (1-indexed)
        
    Returns:
        List of hidden states tensors, one per layer depth
    """
    # Base SiT path.
    if hasattr(model, "forward_features"):
        return model.forward_features(x, t, y, encoder_depths=layer_depths, proj=False)

    # Fallback for SiTWithEncoderKV which does not implement forward_features.
    required_attrs = ("x_embedder", "pos_embed", "t_embedder", "y_embedder", "blocks")
    if not all(hasattr(model, attr) for attr in required_attrs):
        raise AttributeError(
            "Model does not expose forward_features and missing required attrs "
            f"for fallback extraction: {required_attrs}"
        )

    x_tokens = model.x_embedder(x) + model.pos_embed
    t_embed = model.t_embedder(t)
    y_embed = model.y_embedder(y, model.training)
    c = t_embed + y_embed

    hidden_states = []
    max_depth = max(layer_depths)
    requested_depths = set(layer_depths)

    for i, block in enumerate(model.blocks):
        block_out = block(x_tokens, c)
        x_tokens = block_out[0] if isinstance(block_out, tuple) else block_out

        if (i + 1) in requested_depths:
            hidden_states.append(x_tokens)
        if (i + 1) >= max_depth:
            break

    return hidden_states


def evaluate_spatial_metrics(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    num_samples: int = 256,
    timesteps: List[float] = [0.1, 0.5, 0.9],
    layer_depths: List[int] = [4, 8, 12],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Evaluate spatial metrics on DiT hidden states.
    
    Returns:
        Nested dict: {timestep: {layer: {metric: mean_value}}}
    """
    results = {}
    
    # Collect samples
    all_latents = []
    all_labels = []
    count = 0
    
    for batch in tqdm(dataloader, desc="Collecting samples"):
        if len(batch) == 3:
            _, latents, labels = batch
        else:
            latents, labels = batch
        
        latents = latents.squeeze(1).to(device)
        labels = labels.to(device)
        
        # Sample from VAE posterior
        latents = sample_posterior(latents, latents_scale, latents_bias)
        
        all_latents.append(latents)
        all_labels.append(labels)
        count += latents.size(0)
        
        if count >= num_samples:
            break
    
    all_latents = torch.cat(all_latents, dim=0)[:num_samples]
    all_labels = torch.cat(all_labels, dim=0)[:num_samples]
    
    print(f"Collected {all_latents.size(0)} samples")
    
    # Grid size (32x32 / 2x2 patch = 16x16)
    H = W = 16
    
    # Evaluate at different timesteps
    for t_val in timesteps:
        t_key = f"t={t_val:.1f}"
        results[t_key] = {}
        
        # Create noisy latents at this timestep
        t = torch.full((all_latents.size(0),), t_val, device=device)
        noise = torch.randn_like(all_latents)
        
        # Linear interpolation: x_t = (1-t) * x_0 + t * noise
        x_t = (1 - t.view(-1, 1, 1, 1)) * all_latents + t.view(-1, 1, 1, 1) * noise
        
        # Extract hidden states at specified layers
        with torch.no_grad():
            hidden_states_list = extract_hidden_states(
                model, x_t, t, all_labels, layer_depths
            )
        
        # Compute metrics for each layer
        for layer_idx, hidden_states in zip(layer_depths, hidden_states_list):
            layer_key = f"layer_{layer_idx}"
            
            metrics = compute_spatial_metrics(hidden_states, H, W)
            
            results[t_key][layer_key] = {
                'lds': metrics['lds'].mean().item(),
                'cds': metrics['cds'].mean().item(),
                'rmsc': metrics['rmsc'].mean().item(),
            }
    
    return results


def print_results(results: Dict, model_name: str = ""):
    """Pretty print the results."""
    print("\n" + "=" * 70)
    print(f"Spatial Metrics Results {f'({model_name})' if model_name else ''}")
    print("=" * 70)
    
    # Get all timesteps and layers
    timesteps = sorted(results.keys())
    layers = sorted(list(results[timesteps[0]].keys()))
    
    # Print table header
    header = f"{'Timestep':<12} {'Layer':<12} {'LDS':>10} {'CDS':>10} {'RMSC':>10}"
    print(header)
    print("-" * 70)
    
    for t_key in timesteps:
        for layer_key in layers:
            metrics = results[t_key][layer_key]
            print(f"{t_key:<12} {layer_key:<12} {metrics['lds']:>10.4f} {metrics['cds']:>10.4f} {metrics['rmsc']:>10.4f}")
        print()
    
    # Print summary (average across timesteps for final layer)
    final_layer = layers[-1]
    avg_lds = sum(results[t][final_layer]['lds'] for t in timesteps) / len(timesteps)
    avg_cds = sum(results[t][final_layer]['cds'] for t in timesteps) / len(timesteps)
    avg_rmsc = sum(results[t][final_layer]['rmsc'] for t in timesteps) / len(timesteps)
    
    print("-" * 70)
    print(f"{'Average':<12} {final_layer:<12} {avg_lds:>10.4f} {avg_cds:>10.4f} {avg_rmsc:>10.4f}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Compute spatial metrics on DiT hidden states')
    
    # Model/checkpoint
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='../data',
                        help='Path to data directory')
    parser.add_argument('--resolution', type=int, choices=[256, 512], default=256,
                        help='Image resolution for ImageFolder fallback dataset')
    parser.add_argument('--num-samples', type=int, default=256,
                        help='Number of samples to evaluate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for data loading')
    parser.add_argument('--num-workers', type=int, default=12,
                        help='Number of DataLoader workers')
    
    # Evaluation settings
    parser.add_argument('--timesteps', type=str, default='0.1,0.5,0.9',
                        help='Comma-separated timesteps to evaluate')
    parser.add_argument('--layer-depths', type=str, default='4,8,12',
                        help='Comma-separated layer depths (1-indexed)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    
    args = parser.parse_args()
    
    device = args.device
    
    # Parse timesteps and layer depths
    timesteps = [float(t.strip()) for t in args.timesteps.split(',')]
    layer_depths = [int(d.strip()) for d in args.layer_depths.split(',')]
    
    # Load model
    model, ckpt_args = load_model_from_checkpoint(args.checkpoint, device)
    model_name = getattr(ckpt_args, 'exp_name', 'unknown')
    
    # Load VAE latent stats
    latents_stats = torch.load("pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt", 
                                map_location=device, weights_only=False)
    latents_scale = latents_stats['latents_scale'].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats['latents_bias'].to(device).view(1, -1, 1, 1)
    
    # Load dataset
    try:
        dataset = HFImgLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, split="train")
    except Exception as e:
        print(f"Error loading HFImgLatentDataset: {e}")
        print("Falling back to ImageFolderLatentDataset")
        dataset = ImageFolderLatentDataset("sdvae-ft-mse-f8d4", args.data_dir, 
                                            resolution=args.resolution, split="train")
    print(dataset)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    print(f"\nEvaluating model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Timesteps: {timesteps}")
    print(f"Layer depths: {layer_depths}")
    print(f"Num samples: {args.num_samples}")
    
    # Run evaluation
    results = evaluate_spatial_metrics(
        model=model,
        dataloader=dataloader,
        device=device,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        num_samples=args.num_samples,
        timesteps=timesteps,
        layer_depths=layer_depths,
    )
    
    # Print results
    print_results(results, model_name)
    
    # Save results
    output_dir = Path(args.checkpoint).parent.parent / "spatial_metrics"
    output_dir.mkdir(exist_ok=True)
    
    import json
    output_file = output_dir / f"spatial_metrics_{Path(args.checkpoint).stem}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
