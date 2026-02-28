#!/usr/bin/env python3
"""
Attention Entropy over Training — Paper Figure Script.

Computes mean Shannon entropy of attention weights for SiT checkpoints,
following the protocol established by Zhai et al. (ICML 2023):
  - Per-row Shannon entropy: Ent(A_i) = -Σ_j A_{i,j} · log(A_{i,j})
  - Aggregation:  per-token → per-head → per-layer → per-sample → mean
  - Uses natural log (nats);  eps = 1e-8 before log
  - Normalized entropy: H / log(N),  N = sequence length,  range [0,1]

All results are conditioned on the denoising timestep, since attention
patterns differ dramatically between high-noise and low-noise steps.

Three methods compared side-by-side:
  Vanilla SiT  /  iREPA  /  Ours (Scaffolding)

Outputs
-------
1. Training-dynamics curves  (normalized entropy vs. step,
   one panel per layer, separate curves per method)
2. Layer × timestep heatmap  (for selected checkpoints / methods)

Key improvements over v1
------------------------
- Single forward pass extracts ALL requested layers (28× speedup)
- Reports normalized entropy H/log(N) ∈ [0,1]
- Tracks per-head entropy std for error bars / head specialization
- Timestep-group support (low/mid/high noise)
- Proper ImageNet latent loading matching train.py pipeline

Usage
-----
# Full comparison across checkpoints, real data
python compute_entropy.py \
    --methods "SiT-XL/2:/path/to/vanilla/ckpts" \
              "SiT-XL/2+iREPA:/path/to/irepa/ckpts" \
              "SiT-XL/2+Ours:/path/to/ours/ckpts" \
    --stage1-steps 30000 \
    --stage1-methods "SiT-XL/2+Ours" \
    --enc-type dinov2-b \
    --sit-scaffold-layer 4 \
    --enc-scaffold-layer 12 \
    --model SiT-XL/2 \
    --layers 4,14,24 \
    --timesteps 0.1,0.5,0.8 \
    --n-samples 1000 \
    --data-dir /dev/shm/data/ \
    --out entropy_over_training.pdf

# Quick sanity check (random init only, random latents)
python compute_entropy.py \
    --model SiT-XL/2 --layers 4,14,24 \
    --out /tmp/entropy_test.pdf
"""

import argparse
import glob
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

# ── SiT imports ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sit import SiT_models
from models.sit_encoder import SiT_EncoderKV_models


# ═══════════════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════════════

EPS = 1e-8          # Standard epsilon for numerical stability (per survey)
VAE_NAME = "sdvae-ft-mse-f8d4"
LATENTS_STATS_PATH = "pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt"


# ═══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════════════════

def make_fresh_model(model_name: str, device: torch.device,
                     resolution: int = 256) -> torch.nn.Module:
    """Create a freshly-initialized SiT (step 0, random weights)."""
    latent_size = resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        use_cfg=False,
        eval_mode=True,
        **block_kwargs,
    )
    model.eval().to(device)
    return model


def load_model_from_ckpt(ckpt_path: str, model_name: str,
                         device: torch.device,
                         resolution: int = 256) -> torch.nn.Module:
    """Load a SiT model from any checkpoint (vanilla / REPA / Scaffolding)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    latent_size = resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": False}
    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        use_cfg=False,
        eval_mode=True,
        **block_kwargs,
    )

    # Prefer EMA weights
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Filter out extra keys (kv_proj, projectors, etc.)
    filtered = {k: v for k, v in state_dict.items()
                if k in model.state_dict()}
    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    return model


def step_from_filename(path: str) -> int:
    """Extract the training step from a checkpoint filename like 0010000.pt."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading — matches train.py pipeline exactly
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_posterior(moments, latents_scale=1., latents_bias=0.):
    """
    Exact copy of train.py sample_posterior.
    latents are stored as VAE encoder moments (mean, logvar concatenated).
    We sample z = mean + std * noise, then normalise.
    """
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = (z - latents_bias) * latents_scale
    return z


def load_real_latents(data_dir: str, n_samples: int, device: torch.device,
                      latent_size: int = 32):
    """
    Load precomputed latents using the same HFLatentDataset & normalisation
    pipeline as train.py.

    Returns (latents, labels) — latents are normalised and ready for the model.
    """
    from dataset import HFLatentDataset

    dataset = HFLatentDataset(VAE_NAME, data_dir, split="train")

    # Subsample deterministically
    indices = list(range(min(n_samples, len(dataset))))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=min(64, n_samples),
                        shuffle=False, num_workers=4, pin_memory=True)

    # Load latent normalisation stats (same as train.py L361-363)
    latents_stats = torch.load(
        LATENTS_STATS_PATH,
        map_location=device, weights_only=False,
    )
    latents_scale = latents_stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias = latents_stats["latents_bias"].to(device).view(1, -1, 1, 1)

    all_x, all_y = [], []
    for batch in loader:
        x_raw, y = batch
        x_raw = x_raw.squeeze(dim=1).to(device)
        y = y.to(device)
        # Normalise exactly as training: sample_posterior
        x = sample_posterior(x_raw, latents_scale=latents_scale,
                             latents_bias=latents_bias)
        all_x.append(x)
        all_y.append(y)

    return torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)


def make_random_latents(n_samples: int, device: torch.device,
                        latent_size: int = 32):
    """Generate random Gaussian latents (no dataset needed)."""
    x = torch.randn(n_samples, 4, latent_size, latent_size, device=device)
    y = torch.randint(0, 1000, (n_samples,), device=device)
    return x, y


# ═══════════════════════════════════════════════════════════════════════════════
#  Entropy computation  —  ALL layers in ONE forward pass
# ═══════════════════════════════════════════════════════════════════════════════

def extract_attn_weights_all_layers(
    model, x_latent: torch.Tensor,
    t: torch.Tensor, y: torch.Tensor,
    layer_indices: list[int],
) -> dict[int, torch.Tensor]:
    """
    Hook into ALL specified transformer blocks in a SINGLE forward pass
    and capture post-softmax attention weights.

    Returns
    -------
    attn_dict : {layer_idx: Tensor (B, H, N, N)}   — each row sums to 1
    """
    captured: dict[int, torch.Tensor] = {}

    # Patch all target layers at once
    originals = {}
    for li in layer_indices:
        target_block = model.blocks[li]
        attn_module = target_block.attn
        originals[li] = attn_module.forward

        # Closure to capture layer index
        def _make_patched(attn_mod, layer_id):
            def _patched_forward(x_in, attn_mask=None):
                B, N, C = x_in.shape
                qkv = attn_mod.qkv(x_in).reshape(
                    B, N, 3, attn_mod.num_heads, attn_mod.head_dim
                ).permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                q, k = attn_mod.q_norm(q), attn_mod.k_norm(k)
                q = q * attn_mod.scale
                attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # (B, H, N, N)
                captured[layer_id] = attn.detach()

                x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                if hasattr(attn_mod, "norm"):
                    x_out = attn_mod.norm(x_out)
                x_out = attn_mod.proj(x_out)
                if hasattr(attn_mod, "proj_drop"):
                    x_out = attn_mod.proj_drop(x_out)
                return x_out
            return _patched_forward

        attn_module.forward = _make_patched(attn_module, li)

    # Single forward pass
    try:
        with torch.no_grad():
            model(x_latent, t, y)
    finally:
        # Restore originals
        for li in layer_indices:
            model.blocks[li].attn.forward = originals[li]

    return captured


def compute_entropy_stats(attn: torch.Tensor) -> tuple[float, float, float]:
    """
    Shannon entropy in nats (natural log), following Zhai et al.

    Aggregation: per-token → per-head → per-sample → mean.
      H_token = -Σ_j  A_{i,j} · log(A_{i,j} + eps)   for each query i
      H_head  = mean over query tokens
      H_sample = per-head values  (B, H)  — keep head dim for std

    Parameters
    ----------
    attn : (B, H, N, N)  — each row sums to 1.

    Returns
    -------
    mean_entropy : Scalar mean entropy in nats.
    head_std     : Std of per-head mean entropy (measures head specialisation).
    max_entropy  : log(N) for reference.
    """
    N = attn.shape[-1]
    H_per_token = -(attn * (attn + EPS).log()).sum(dim=-1)   # (B, H, N)
    H_per_head = H_per_token.mean(dim=-1)                     # (B, H)
    # Mean over heads, then batch
    H_per_sample = H_per_head.mean(dim=-1)                    # (B,)
    mean_ent = H_per_sample.mean().item()

    # Std across heads (mean over batch first, then std over heads)
    head_means = H_per_head.mean(dim=0)                       # (H,)
    head_std = head_means.std().item()

    max_ent = math.log(N)
    return mean_ent, head_std, max_ent


def compute_entropy_for_model(
    model,
    layer_indices: list[int],
    all_x: torch.Tensor,
    all_y: torch.Tensor,
    timestep: float,
    batch_size: int = 32,
) -> dict[int, tuple[float, float]]:
    """
    Compute mean attention entropy (nats) and per-head std for each layer
    at a given timestep.  ALL layers extracted in ONE forward pass per batch.

    Returns: {layer_idx: (mean_entropy, head_std)}
    """
    device = next(model.parameters()).device
    n = all_x.shape[0]
    n_batches = math.ceil(n / batch_size)

    # Accumulators:  layer → (sum_ent, sum_std, count)
    acc_ent = {li: 0.0 for li in layer_indices}
    acc_std = {li: 0.0 for li in layer_indices}

    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, n)
        x = all_x[start:end].to(device)
        y = all_y[start:end].to(device)
        bs = x.shape[0]
        t = torch.full((bs,), timestep, device=device, dtype=torch.float32)

        # One forward pass → all layers
        attn_dict = extract_attn_weights_all_layers(model, x, t, y, layer_indices)

        for li in layer_indices:
            mean_ent, head_std, _ = compute_entropy_stats(attn_dict[li])
            acc_ent[li] += mean_ent * bs
            acc_std[li] += head_std * bs

    return {li: (acc_ent[li] / n, acc_std[li] / n) for li in layer_indices}


# ═══════════════════════════════════════════════════════════════════════════════
#  Stage-1 support  (DINO-KV injection at scaffold layer)
# ═══════════════════════════════════════════════════════════════════════════════

def load_real_data_with_images(data_dir: str, n_samples: int, device: torch.device,
                                latent_size: int = 32):
    """
    Load pixel-space images AND precomputed latents using HFImgLatentDataset.
    Returns (all_img, all_x, all_y).
      all_img : (N, C, H, W) uint8  [0, 255]
      all_x   : (N, 4, latent_size, latent_size) normalised latents
      all_y   : (N,) class labels
    """
    from dataset import HFImgLatentDataset
    dataset = HFImgLatentDataset(VAE_NAME, data_dir, split="train")
    indices = list(range(min(n_samples, len(dataset))))
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=min(64, n_samples),
                         shuffle=False, num_workers=4, pin_memory=True)

    latents_stats = torch.load(LATENTS_STATS_PATH, map_location=device,
                                weights_only=False)
    latents_scale = latents_stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias  = latents_stats["latents_bias"].to(device).view(1, -1, 1, 1)

    all_img, all_x, all_y = [], [], []
    for imgs, latents, labels in loader:
        imgs    = imgs.to(device)
        latents = latents.squeeze(1).to(device)
        labels  = labels.to(device)
        x = sample_posterior(latents, latents_scale=latents_scale,
                             latents_bias=latents_bias)
        all_img.append(imgs)
        all_x.append(x)
        all_y.append(labels)

    return torch.cat(all_img), torch.cat(all_x), torch.cat(all_y)


def load_encoder_for_stage1(enc_type: str, enc_layer_idx: int,
                             device: torch.device, resolution: int = 256):
    """
    Load DINO/DINOv2 encoder and wire an EncoderKVExtractor on the
    requested (0-based) encoder layer.  Returns (encoder, extractor).
    """
    from vision_encoder import load_encoders
    from models.encoder_adapter import EncoderKVExtractor

    encoders  = load_encoders(enc_type, device, resolution)
    encoder   = encoders[0]
    encoder.eval()
    extractor = EncoderKVExtractor(encoder.model, layer_indices=[enc_layer_idx])
    return encoder, extractor


def load_model_enc_kv(ckpt_path: str, device: torch.device,
                      resolution: int = 256) -> torch.nn.Module:
    """
    Load the full SiT_EncoderKV model from a Stage-1 checkpoint.
    Constructor args are inferred from ckpt['args'].
    fused_attn is forced to False so our hook can materialise Q/K/V.
    """
    ckpt      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)
    latent_size = resolution // 8

    def _get(attr, default=None):
        return getattr(ckpt_args, attr, default) if ckpt_args else default

    def _parse_1based(val):
        """Convert '4' or [4] (1-based) to a 0-based list."""
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return [int(x) - 1 for x in val]
        return [int(x) - 1 for x in str(val).split(",")]

    mk = dict(
        input_size=latent_size, in_channels=4,
        num_classes=1000, use_cfg=False,
        eval_mode=True, fused_attn=False, qk_norm=False,
    )
    for ck_attr, mk_key in [
        ("enc_dim",         "enc_dim"),
        ("enc_heads",       "enc_heads"),
        ("kv_proj_type",    "kv_proj_type"),
        ("kv_norm_type",    "kv_norm_type"),
        ("kv_zscore_alpha", "kv_zscore_alpha"),
        ("kv_replace_mode", "kv_replace_mode"),
    ]:
        v = _get(ck_attr)
        if v is not None:
            mk[mk_key] = v

    sit_idx = _parse_1based(_get("sit_layer_indices"))
    enc_idx = _parse_1based(_get("enc_layer_indices"))
    if sit_idx is not None:
        mk["sit_layer_indices"] = sit_idx
    if enc_idx is not None:
        mk["enc_layer_indices"] = enc_idx

    model_name = _get("model", "SiT-XL/2-EncoderKV")
    model = SiT_EncoderKV_models[model_name](**mk)

    if "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    filtered = {k: v for k, v in state_dict.items() if k in model.state_dict()}
    model.load_state_dict(filtered, strict=False)
    model.eval().to(device)
    return model


@torch.no_grad()
def extract_attn_stage1(
    model_enc,
    encoder,              # VisionEncoder wrapper (.preprocess + .model)
    enc_extractor,        # EncoderKVExtractor
    enc_layer_idx: int,   # 0-based encoder layer we hooked
    scaffold_sit_idx: int,# 0-based SiT layer where DINO KV is injected
    all_img: torch.Tensor,# (N, C, H, W) uint8 [0, 255]
    all_x:   torch.Tensor,# (N, 4, H, W) normalised latents
    all_y:   torch.Tensor,# (N,) labels
    timestep: float,
    layer_indices: list,
    batch_size: int = 32,
) -> dict:
    """
    Compute per-layer attention entropy for a Stage-1 checkpoint.

    Scaffold layer  → Q_SiT @ K_DINO  (DINO KV projected via kv_proj)
    All other layers → Q_SiT @ K_SiT  (native SiT KV)

    Returns {layer_idx: (mean_entropy_nats, head_std)}.
    """
    device  = next(model_enc.parameters()).device
    n       = all_x.shape[0]
    acc_ent = {li: 0.0 for li in layer_indices}
    acc_std = {li: 0.0 for li in layer_indices}

    scaffold_block = model_enc.blocks[scaffold_sit_idx]

    for b_start in range(0, n, batch_size):
        b_end = min(b_start + batch_size, n)
        img = all_img[b_start:b_end].to(device)   # uint8
        x   = all_x  [b_start:b_end].to(device)
        y   = all_y  [b_start:b_end].to(device)
        bs  = x.shape[0]
        t   = torch.full((bs,), timestep, device=device, dtype=torch.float32)

        # 1. Run DINO to get raw KV from the hooked encoder layer
        img_enc = encoder.preprocess(img)
        enc_extractor.reset_cache()
        encoder.model(img_enc)
        q_raw, k_raw, v_raw = enc_extractor.captured_kv[enc_layer_idx]

        # 2. Project DINO KV into SiT head space via kv_proj
        _q_proj, k_proj, v_proj = scaffold_block.kv_proj(
            q_enc=q_raw, k_enc=k_raw, v_enc=v_raw, stage=1
        )
        # k_proj, v_proj: (B, sit_heads, N_enc, sit_head_dim)

        # 3. Hook all target attention modules
        captured  = {}
        originals = {}

        for li in layer_indices:
            attn_mod      = model_enc.blocks[li].attn
            originals[li] = attn_mod.forward

            def _make_hook(layer_id, _k=None, _v=None):
                # AttentionWithEncoderKV.forward returns (x_out, distill_loss)
                def _hook(x_in, q_enc=None, k_enc=None, v_enc=None,
                          stage=2, align_mode="attn_mse",
                          time_input=None, path_type="linear",
                          kv_replace_mode="kv"):
                    B, N, C = x_in.shape
                    qkv = attn_mod.qkv(x_in).reshape(
                        B, N, 3, attn_mod.num_heads, attn_mod.head_dim
                    ).permute(2, 0, 3, 1, 4)
                    q_sit, k_sit, v_sit = qkv.unbind(0)
                    q_sit = attn_mod.q_norm(q_sit)
                    k_sit = attn_mod.k_norm(k_sit)

                    k = _k if _k is not None else k_sit
                    v = _v if _v is not None else v_sit

                    attn = (q_sit * attn_mod.scale @ k.transpose(-2, -1)).softmax(-1)
                    captured[layer_id] = attn.detach()
                    x_out = (attn @ v).transpose(1, 2).reshape(B, N, C)
                    x_out = attn_mod.proj(x_out)
                    return x_out, None   # mimic (output, distill_loss)
                return _hook

            is_scaffold = (li == scaffold_sit_idx)
            attn_mod.forward = _make_hook(
                li,
                _k=k_proj if is_scaffold else None,
                _v=v_proj if is_scaffold else None,
            )

        # 4. Single forward pass – enc_kv_list=None, so block skips kv_proj internally
        try:
            with torch.no_grad():
                model_enc(x, t, y)
        finally:
            for li in layer_indices:
                model_enc.blocks[li].attn.forward = originals[li]

        for li in layer_indices:
            mean_ent, head_std, _ = compute_entropy_stats(captured[li])
            acc_ent[li] += mean_ent * bs
            acc_std[li] += head_std * bs

    return {li: (acc_ent[li] / n, acc_std[li] / n) for li in layer_indices}


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint discovery
# ═══════════════════════════════════════════════════════════════════════════════

def discover_checkpoints(path: str, max_step: int | None = None,
                         steps: list[int] | None = None) -> list[tuple[int, str]]:
    """
    Given a path (directory or single .pt), return a sorted list of
    (step, filepath) pairs.

    If `steps` is given, keep only the checkpoint whose step is closest
    to each requested step (useful for sparse evaluation like 10K,20K,...).
    """
    if os.path.isfile(path) and path.endswith(".pt"):
        return [(step_from_filename(path), path)]

    if os.path.isdir(path):
        pts = sorted(glob.glob(os.path.join(path, "*.pt")))
        pairs = [(step_from_filename(p), p) for p in pts]
        if max_step is not None:
            pairs = [(s, p) for s, p in pairs if s <= max_step]
        pairs = sorted(pairs, key=lambda x: x[0])

        if steps is not None:
            # For each requested step, pick the closest available checkpoint
            selected = {}
            for target in steps:
                best = min(pairs, key=lambda x: abs(x[0] - target))
                selected[best[0]] = best[1]  # deduplicate by actual step
            pairs = sorted(selected.items(), key=lambda x: x[0])

        return pairs

    raise FileNotFoundError(f"Not a file or directory: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════════════

# Publication-friendly palette
METHOD_STYLES = {
    "Vanilla":     {"color": "#8B8B8B", "marker": "o", "linestyle": "-"},
    "iREPA":       {"color": "#4C9BE8", "marker": "s", "linestyle": "--"},
    "REPA":        {"color": "#4C9BE8", "marker": "s", "linestyle": "--"},
    "Ours":        {"color": "#E8524C", "marker": "D", "linestyle": "-"},
    "Scaffolding": {"color": "#E8524C", "marker": "D", "linestyle": "-"},
}
DEFAULT_STYLE = {"color": "#6A6A6A", "marker": "^", "linestyle": "-."}


def plot_training_curves(
    data: dict,           # {method: {(layer, timestep): [(step, ent, std), ...]}}
    layer_indices: list[int],
    timesteps: list[float],
    n_tokens: int,
    save_path: str,
    use_normalized: bool = True,
    stage1_steps: int | None = None,   # draw vertical boundary line if set
):
    """
    Training-dynamics plot: entropy vs. step.

    Layout: rows = timesteps,  cols = layers.
    If only one timestep, just one row.
    """
    max_entropy = math.log(n_tokens)  # nats (ln)

    n_rows = len(timesteps)
    n_cols = len(layer_indices)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.2 * n_cols, 3.5 * n_rows),
                             sharex=True, sharey=True, squeeze=False)

    for ri, ts in enumerate(timesteps):
        for ci, li in enumerate(layer_indices):
            ax = axes[ri][ci]

            if use_normalized:
                # Reference line at 1.0 (uniform)
                ax.axhline(1.0, color="gray", linestyle=":",
                           linewidth=1.0, alpha=0.7)
                if ri == 0 and ci == 0:
                    ax.text(0.02, 1.0, " uniform",
                            transform=ax.get_yaxis_transform(),
                            fontsize=8, color="gray", va="bottom")
            else:
                ax.axhline(max_entropy, color="gray", linestyle=":",
                           linewidth=1.0, alpha=0.7)
                if ri == 0 and ci == 0:
                    ax.text(0.02, max_entropy, f" ln({n_tokens})",
                            transform=ax.get_yaxis_transform(),
                            fontsize=8, color="gray", va="bottom")

            # Stage-boundary vertical line
            if stage1_steps is not None:
                ax.axvline(stage1_steps, color="#AA6600", linestyle="--",
                           linewidth=1.2, alpha=0.8)
                if ri == 0 and ci == 0:
                    ax.text(stage1_steps, 1.06, " S1→S2",
                            transform=ax.get_xaxis_transform(),
                            fontsize=7, color="#AA6600", va="top")

            for method, method_data in data.items():
                key = (li, ts)
                if key not in method_data:
                    continue
                pts = sorted(method_data[key], key=lambda x: x[0])
                steps = [s for s, _, _ in pts]
                if use_normalized:
                    ents = [e / max_entropy for _, e, _ in pts]
                    stds = [sd / max_entropy for _, _, sd in pts]
                else:
                    ents = [e for _, e, _ in pts]
                    stds = [sd for _, _, sd in pts]

                style = METHOD_STYLES.get(method, DEFAULT_STYLE)
                ax.plot(steps, ents, label=method,
                        **style, markersize=4, linewidth=1.8, alpha=0.9)
                # Shaded band: ±1 head-std
                if any(sd > 0 for sd in stds):
                    lower = [e - s for e, s in zip(ents, stds)]
                    upper = [e + s for e, s in zip(ents, stds)]
                    ax.fill_between(steps, lower, upper,
                                    color=style["color"], alpha=0.12)

            # Labels
            if ri == 0:
                ax.set_title(f"Layer {li + 1}", fontsize=11, fontweight="bold")
            if ri == n_rows - 1:
                ax.set_xlabel("Training Step", fontsize=10)
            if ci == 0:
                label = f"t = {ts}" if not isinstance(ts, str) else ts
                ax.set_ylabel(label, fontsize=10, fontweight="bold")

            ax.xaxis.set_major_formatter(ticker.FuncFormatter(
                lambda x, _: f"{int(x / 1000)}K" if x >= 1000 else str(int(x))))
            ax.grid(True, alpha=0.2)
            ax.set_ylim(bottom=0)
            if use_normalized:
                ax.set_ylim(0, 1.1)
                ax.set_ylabel(
                    (f"$t = {ts}$\n" if ci == 0 else "") +
                    r"$H / \log N$",
                    fontsize=10)

    # Single legend at top
    handles, labels = axes[0][0].get_legend_handles_labels()
    seen = set()
    unique_h, unique_l = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_h.append(h)
            unique_l.append(l)
    fig.legend(unique_h, unique_l, loc="upper center",
               ncol=len(unique_l), fontsize=10,
               bbox_to_anchor=(0.5, 1.04), frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved training curves → {save_path}")


def plot_layer_timestep_heatmap(
    heatmap_data: dict,    # {method: np.array of shape (n_layers, n_timesteps)}
    layer_indices: list[int],
    timesteps: list[float],
    n_tokens: int,
    save_path: str,
    use_normalized: bool = True,
):
    """
    Layer × timestep heatmap for each method, side by side.
    """
    max_entropy = math.log(n_tokens)
    methods = list(heatmap_data.keys())
    n_methods = len(methods)

    fig, axes = plt.subplots(1, n_methods,
                             figsize=(3.8 * n_methods + 0.8, 3.5),
                             squeeze=False)

    if use_normalized:
        vmin, vmax = 0, 1.0
    else:
        vmin, vmax = 0, max_entropy

    for mi, method in enumerate(methods):
        ax = axes[0][mi]
        mat = heatmap_data[method]  # (n_layers, n_timesteps)
        if use_normalized:
            mat = mat / max_entropy

        im = ax.imshow(mat, aspect="auto", cmap="RdYlBu_r",
                       vmin=vmin, vmax=vmax,
                       interpolation="nearest", origin="lower")

        # Axes
        ax.set_xticks(range(len(timesteps)))
        ax.set_xticklabels([f"{t:.1f}" for t in timesteps], fontsize=8)
        ax.set_yticks(range(len(layer_indices)))
        ax.set_yticklabels([f"L{li + 1}" for li in layer_indices], fontsize=8)
        ax.set_xlabel("Timestep $t$", fontsize=10)
        if mi == 0:
            ax.set_ylabel("Layer", fontsize=10)
        ax.set_title(method, fontsize=11, fontweight="bold")

        # Annotate cells with values
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                text_color = "white" if val > 0.6 * vmax else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=text_color)

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.03)
    cbar_label = r"$H / \log N$" if use_normalized else "Entropy (nats)"
    cbar.set_label(cbar_label, fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(f"Saved heatmap → {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_method_arg(s: str) -> tuple[str, str]:
    """Parse 'Label:/path/to/checkpoints' into (label, path)."""
    if ":" in s and not s.startswith("/"):
        label, path = s.split(":", 1)
        return label.strip(), path.strip()
    return os.path.basename(s.rstrip("/")), s.strip()


def main():
    parser = argparse.ArgumentParser(
        description="Compute attention entropy over SiT training checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--methods", type=parse_method_arg, nargs="*", default=[],
        help='One or more "Label:/path/to/ckpt_dir_or_file" entries')
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()))
    parser.add_argument("--layers", type=str, default="4,14,24",
                        help="Comma-separated 1-based layer indices "
                             "(use 'all' for all 28 layers)")
    parser.add_argument("--timesteps", type=str, default="0.1,0.3,0.5,0.7,0.9",
                        help="Comma-separated timesteps to evaluate")
    parser.add_argument("--n-samples", type=int, default=1000,
                        help="Number of samples for entropy averaging (≥512 recommended)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for forward passes")
    parser.add_argument("--max-step", type=int, default=None,
                        help="Only include checkpoints up to this step")
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated list of training steps to evaluate, "
                             "e.g. '10000,20000,50000,100000'. Picks nearest checkpoint.")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data dir with precomputed latents. "
                             "If not set, uses random Gaussian latents.")
    parser.add_argument("--resolution", type=int, default=256,
                        choices=[256, 512])
    parser.add_argument("--out", type=str, default="entropy_over_training.pdf")
    parser.add_argument("--out-heatmap", type=str, default=None,
                        help="Output path for layer×timestep heatmap (default: auto)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", action="store_true",
                        help="Save raw results to CSV alongside figures")
    # Stage-1 / scaffold options
    parser.add_argument("--stage1-steps", type=int, default=None,
                        help="Checkpoints with step ≤ this value are measured in "
                             "Stage-1 mode (DINO KV at scaffold layer). "
                             "Requires --enc-type and real --data-dir.")
    parser.add_argument("--stage1-methods", type=str, default=None,
                        help="Comma-separated method labels that use Stage-1 DINO KV "
                             "(e.g. 'Ours'). Other methods always use their native KV. "
                             "If not set, Stage-1 applies to ALL methods.")
    parser.add_argument("--enc-type", type=str, default="dinov2-b",
                        help="Encoder type for Stage-1 KV (e.g. 'dinov2-b'). "
                             "Passed to load_encoders().")
    parser.add_argument("--sit-scaffold-layer", type=int, default=4,
                        help="1-based SiT block index where DINO KV is injected "
                             "(matches sit-layer-indices in training config).")
    parser.add_argument("--enc-scaffold-layer", type=int, default=12,
                        help="1-based encoder layer from which to extract KV "
                             "(matches enc-layer-indices in training config).")

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Parse params
    latent_size = args.resolution // 8
    patch_size = 2
    n_tokens = (latent_size // patch_size) ** 2
    max_entropy = math.log(n_tokens)

    # Parse layers
    if args.layers.strip().lower() == "all":
        # SiT-XL has 28 blocks
        layer_indices = list(range(28))
    else:
        layer_indices = [int(x) - 1 for x in args.layers.split(",")]

    timesteps = [float(t) for t in args.timesteps.split(",")]
    target_steps = ([int(s) for s in args.steps.split(",")]
                    if args.steps else None)
    # Methods that use Stage-1 DINO-KV measurement (None = all methods)
    stage1_method_set = (set(args.stage1_methods.split(","))
                         if args.stage1_methods else None)

    print(f"Tokens N = {n_tokens}")
    print(f"Max entropy (uniform) = ln({n_tokens}) = {max_entropy:.4f} nats")
    print(f"Layers (1-based): {[li + 1 for li in layer_indices]}")
    print(f"Timesteps: {timesteps}")
    print(f"Samples: {args.n_samples}")

    # ── Stage-1 setup ─────────────────────────────────────────────────────────
    use_stage1  = args.stage1_steps is not None
    all_img     = None   # pixel-space images, only loaded when needed
    encoder_s1  = None   # VisionEncoder for Stage-1
    extractor_s1= None   # EncoderKVExtractor for Stage-1
    sit_scaffold_idx = args.sit_scaffold_layer - 1   # 0-based
    enc_scaffold_idx = args.enc_scaffold_layer - 1   # 0-based

    if use_stage1:
        print(f"\nStage-1 mode enabled: steps ≤ {args.stage1_steps} use DINO KV "
              f"(enc={args.enc_type}, enc_layer={args.enc_scaffold_layer}, "
              f"sit_layer={args.sit_scaffold_layer})")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.data_dir is not None:
        if use_stage1:
            print(f"\nLoading images + latents from {args.data_dir} ...")
            all_img, all_x, all_y = load_real_data_with_images(
                args.data_dir, args.n_samples, device, latent_size)
            print(f"Loaded {all_x.shape[0]} samples — "
                  f"img {all_img.shape}, latent {all_x.shape}")
        else:
            print(f"\nLoading real latents from {args.data_dir} ...")
            all_x, all_y = load_real_latents(
                args.data_dir, args.n_samples, device, latent_size)
            print(f"Loaded {all_x.shape[0]} samples, shape {all_x.shape}")
    else:
        print("\nUsing random Gaussian latents (no --data-dir provided)")
        all_x, all_y = make_random_latents(args.n_samples, device, latent_size)

    # ── Load DINO encoder (once, reused across all checkpoints) ───────────────
    if use_stage1 and all_img is not None:
        print(f"\nLoading encoder '{args.enc_type}' for Stage-1 measurement ...")
        encoder_s1, extractor_s1 = load_encoder_for_stage1(
            args.enc_type, enc_scaffold_idx, device, args.resolution)
        print("Encoder loaded.")

    # ── Step 0: random-init baseline ──────────────────────────────────────────
    print("\n─── Step 0: random initialization ───")
    fresh = make_fresh_model(args.model, device, args.resolution)

    init_entropy: dict[tuple[int, float], tuple[float, float]] = {}
    for ts in timesteps:
        results = compute_entropy_for_model(
            fresh, layer_indices, all_x, all_y, timestep=ts,
            batch_size=args.batch_size,
        )
        for li, (ent, hstd) in results.items():
            init_entropy[(li, ts)] = (ent, hstd)
            norm_ent = ent / max_entropy
            print(f"  t={ts:.1f}  Layer {li + 1:2d}: H = {ent:.4f} nats  "
                  f"H/logN = {norm_ent:.4f}  head_std = {hstd:.4f}")

    del fresh
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── Collect entropy for each method ───────────────────────────────────────
    # data[method][(layer, timestep)] = [(step, entropy, head_std), ...]
    data: dict[str, dict[tuple[int, float], list[tuple[int, float, float]]]] = {}

    for label, ckpt_path in args.methods:
        print(f"\n─── {label}: {ckpt_path} ───")
        checkpoints = discover_checkpoints(ckpt_path, args.max_step,
                                           steps=target_steps)
        if not checkpoints:
            print(f"  [warn] No checkpoints found, skipping.")
            continue

        # Initialize with step-0 from random init
        method_data: dict[tuple[int, float], list[tuple[int, float, float]]] = {}
        for key, (ent, hstd) in init_entropy.items():
            method_data[key] = [(0, ent, hstd)]

        # Whether this method participates in Stage-1 measurement
        method_uses_stage1 = (use_stage1 and encoder_s1 is not None and
                               (stage1_method_set is None or label in stage1_method_set))

        for step, fpath in checkpoints:
            is_stage1 = (method_uses_stage1 and step <= args.stage1_steps)
            stage_tag = "S1(DINO-KV)" if is_stage1 else "S2(native-KV)"
            print(f"  Step {step:>7d} [{stage_tag}]: {os.path.basename(fpath)} ...")

            if is_stage1:
                model = load_model_enc_kv(fpath, device, args.resolution)
            else:
                model = load_model_from_ckpt(fpath, args.model, device,
                                             args.resolution)

            for ts in timesteps:
                if is_stage1:
                    results = extract_attn_stage1(
                        model, encoder_s1, extractor_s1,
                        enc_scaffold_idx, sit_scaffold_idx,
                        all_img, all_x, all_y,
                        timestep=ts, layer_indices=layer_indices,
                        batch_size=args.batch_size,
                    )
                else:
                    results = compute_entropy_for_model(
                        model, layer_indices, all_x, all_y, timestep=ts,
                        batch_size=args.batch_size,
                    )
                for li, (ent, hstd) in results.items():
                    key = (li, ts)
                    method_data[key].append((step, ent, hstd))

                print(f"    t={ts:.1f}: " + ", ".join(
                    f"L{li+1}={ent:.3f}({hstd:.3f})"
                    for li, (ent, hstd) in results.items()))

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        data[label] = method_data

    if not data:
        print("\nNo methods provided — plotting random-init only.")
        data["Random Init"] = {key: [(0, ent, hstd)]
                                for key, (ent, hstd) in init_entropy.items()}

    # ── Save raw data (CSV) ───────────────────────────────────────────────────
    if args.save_csv:
        csv_path = os.path.splitext(args.out)[0] + ".csv"
        with open(csv_path, "w") as f:
            f.write("method,step,layer,timestep,entropy_nats,normalized_entropy,head_std\n")
            for method, method_data in data.items():
                for (li, ts), pts in method_data.items():
                    for step, ent, hstd in pts:
                        norm = ent / max_entropy
                        f.write(f"{method},{step},{li+1},{ts},{ent:.6f},{norm:.6f},{hstd:.6f}\n")
        print(f"Saved CSV → {csv_path}")

    # ── Plot 1: Training dynamics curves ──────────────────────────────────────
    plot_training_curves(data, layer_indices, timesteps, n_tokens, args.out,
                         use_normalized=True,
                         stage1_steps=args.stage1_steps)

    # ── Plot 2: Layer × timestep heatmap (for the latest checkpoint) ─────────
    heatmap_path = args.out_heatmap
    if heatmap_path is None:
        base, ext = os.path.splitext(args.out)
        heatmap_path = f"{base}_heatmap{ext}"

    heatmap_data = {}
    for method, method_data in data.items():
        mat = np.zeros((len(layer_indices), len(timesteps)))
        for i, li in enumerate(layer_indices):
            for j, ts in enumerate(timesteps):
                key = (li, ts)
                if key in method_data and method_data[key]:
                    # Use the latest step's entropy
                    pts = sorted(method_data[key], key=lambda x: x[0])
                    mat[i, j] = pts[-1][1]  # entropy (nats)
        heatmap_data[method] = mat

    plot_layer_timestep_heatmap(
        heatmap_data, layer_indices, timesteps, n_tokens, heatmap_path,
        use_normalized=True)

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("Summary: Entropy at step 0 (random init)")
    print(f"{'Layer':>8s}", end="")
    for ts in timesteps:
        print(f"  t={ts:.1f} (H/logN)", end="")
    print()
    print("-" * 80)
    for li in layer_indices:
        print(f"  L{li+1:>4d}  ", end="")
        for ts in timesteps:
            ent, hstd = init_entropy[(li, ts)]
            norm = ent / max_entropy
            print(f"  {norm:5.3f} ±{hstd/max_entropy:.3f}", end="")
        print()
    print("=" * 80)


if __name__ == "__main__":
    main()
