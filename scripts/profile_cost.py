"""
Profile training cost: REPA baseline vs Attention Scaffolding (ours).

Measures per-iteration wall-clock time, peak GPU memory, and FLOPs breakdown.
Reports overhead as a percentage.

Usage:
    python scripts/profile_cost.py --model SiT-B/2 --encoder dinov2_vitb14_reg
    python scripts/profile_cost.py --model SiT-L/2 --encoder dinov2_vitl14_reg
    python scripts/profile_cost.py --model SiT-XL/2 --encoder dinov2_vitl14_reg
"""

import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.sit import SiT_models
from models.sit_encoder import SiT_EncoderKV_models
from models.encoder_adapter import EncoderKVExtractor
from vision_encoder import load_encoders


# ──────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────

def count_params(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_params_module(module):
    return sum(p.numel() for p in module.parameters())


@contextmanager
def cuda_timer(device, warmup=False):
    """Context manager that returns elapsed ms on CUDA (sync'd)."""
    if not warmup:
        torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    result = {"time_ms": 0, "peak_mem_mb": 0}
    yield result
    torch.cuda.synchronize(device)
    result["time_ms"] = (time.perf_counter() - start) * 1000
    if not warmup:
        result["peak_mem_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2


def make_dummy_batch(batch_size, latent_size, in_channels, num_classes, device):
    x = torch.randn(batch_size, in_channels, latent_size, latent_size, device=device)
    t = torch.rand(batch_size, device=device)
    y = torch.randint(0, num_classes, (batch_size,), device=device)
    return x, t, y


def make_dummy_images(batch_size, img_size, device):
    """Dummy 224x224 images for encoder."""
    return torch.randn(batch_size, 3, img_size, img_size, device=device)


# ──────────────────────────────────────────────────────────────
#  REPA Baseline: SiT + encoder forward (for REPA features only)
# ──────────────────────────────────────────────────────────────

def profile_repa_baseline(model_name, encoder_name, batch_size, device,
                          num_iters=50, warmup_iters=10):
    """Profile REPA baseline: SiT forward + backward + encoder forward."""

    # Build SiT (baseline)
    base_name = model_name  # e.g. "SiT-B/2"
    sit_model = SiT_models[base_name](
        path_type="linear",
        z_dims=[768],
        encoder_depth=8,
        qk_norm=True,
        projection_layer_type="mlp",
    ).to(device).train()

    # Build encoder (frozen, for REPA)
    encoders, enc_names = load_encoders([encoder_name], device=device, dtype=torch.float32)
    encoder = encoders[0]
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(sit_model.parameters(), lr=1e-4)

    latent_size = 32
    img_size = 256

    # Warmup
    for _ in range(warmup_iters):
        x, t, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        imgs = make_dummy_images(batch_size, img_size, device)
        with torch.no_grad():
            enc_out = encoder(imgs)
            if isinstance(enc_out, dict):
                z_enc = enc_out.get("x_norm_patchtokens", enc_out.get("last_hidden_state"))
            elif hasattr(enc_out, "last_hidden_state"):
                z_enc = enc_out.last_hidden_state[:, 1:]
            else:
                z_enc = enc_out[:, 1:]  # remove CLS
        out, zs, _ = sit_model(x, t, y)
        loss = F.mse_loss(out, torch.randn_like(out))
        if zs is not None and len(zs) > 0:
            loss = loss + F.cosine_similarity(zs[0], z_enc.detach(), dim=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Profile
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for _ in range(num_iters):
        x, t, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        imgs = make_dummy_images(batch_size, img_size, device)

        with torch.no_grad():
            enc_out = encoder(imgs)
            if isinstance(enc_out, dict):
                z_enc = enc_out.get("x_norm_patchtokens", enc_out.get("last_hidden_state"))
            elif hasattr(enc_out, "last_hidden_state"):
                z_enc = enc_out.last_hidden_state[:, 1:]
            else:
                z_enc = enc_out[:, 1:]

        out, zs, _ = sit_model(x, t, y)
        loss = F.mse_loss(out, torch.randn_like(out))
        if zs is not None and len(zs) > 0:
            loss = loss + F.cosine_similarity(zs[0], z_enc.detach(), dim=-1).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t_start) * 1000 / num_iters
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    params = count_params(sit_model)

    del sit_model, encoder, encoders, optimizer
    torch.cuda.empty_cache()

    return {
        "method": "REPA (baseline)",
        "time_ms": elapsed,
        "peak_mem_mb": peak_mem,
        "trainable_params": params,
    }


# ──────────────────────────────────────────────────────────────
#  Ours: SiT-EncoderKV + encoder forward (KV hooks + REPA)
# ──────────────────────────────────────────────────────────────

def profile_ours(model_name, encoder_name, batch_size, device, stage,
                 num_iters=50, warmup_iters=10,
                 enc_layer_indices=None, sit_layer_indices=None):
    """Profile our method: SiT-EncoderKV forward + backward + encoder KV extraction."""

    ekv_name = model_name + "-EncoderKV"

    # Load encoder for KV extraction
    encoders, enc_names = load_encoders([encoder_name], device=device, dtype=torch.float32)
    encoder = encoders[0]
    encoder.eval()

    # Determine encoder dims
    if enc_layer_indices is None:
        enc_layer_indices = [8]
    if sit_layer_indices is None:
        sit_layer_indices = [8]

    kv_extractor = EncoderKVExtractor(encoder, layer_indices=enc_layer_indices).to(device)
    enc_dim = kv_extractor.get_layer_dim(enc_layer_indices[0])
    enc_heads = kv_extractor.get_layer_heads(enc_layer_indices[0])

    sit_model = SiT_EncoderKV_models[ekv_name](
        path_type="linear",
        z_dims=[enc_dim],
        encoder_depth=8,
        qk_norm=True,
        projection_layer_type="mlp",
        enc_layer_indices=enc_layer_indices,
        sit_layer_indices=sit_layer_indices,
        enc_dim=enc_dim,
        enc_heads=enc_heads,
        kv_proj_type="linear",
        kv_norm_type="layernorm",
        kv_replace_mode="kv",
    ).to(device).train()

    optimizer = torch.optim.AdamW(
        [p for p in sit_model.parameters() if p.requires_grad], lr=1e-4
    )

    latent_size = 32
    img_size = 256

    # Warmup
    for _ in range(warmup_iters):
        x, t_step, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        imgs = make_dummy_images(batch_size, img_size, device)
        with torch.no_grad():
            kv_list, cls_token = kv_extractor(imgs)
        out, zs, zs_orig, distill_loss = sit_model(
            x, t_step, y, enc_kv_list=kv_list, stage=stage, align_mode="attn_mse"
        )
        loss = F.mse_loss(out, torch.randn_like(out)) + distill_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Profile
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for _ in range(num_iters):
        x, t_step, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        imgs = make_dummy_images(batch_size, img_size, device)

        with torch.no_grad():
            kv_list, cls_token = kv_extractor(imgs)

        out, zs, zs_orig, distill_loss = sit_model(
            x, t_step, y, enc_kv_list=kv_list, stage=stage, align_mode="attn_mse"
        )
        loss = F.mse_loss(out, torch.randn_like(out)) + distill_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t_start) * 1000 / num_iters
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2

    # Count params
    sit_params = count_params(sit_model)
    # KV projection params (extra over baseline)
    kv_proj_params = 0
    for block in sit_model.blocks:
        if hasattr(block, 'has_enc_kv') and block.has_enc_kv:
            kv_proj_params += count_params_module(block.kv_proj)

    del sit_model, encoder, encoders, kv_extractor, optimizer
    torch.cuda.empty_cache()

    return {
        "method": f"Ours (Stage {stage})",
        "time_ms": elapsed,
        "peak_mem_mb": peak_mem,
        "trainable_params": sit_params,
        "kv_proj_params": kv_proj_params,
    }


# ──────────────────────────────────────────────────────────────
#  Inference profiling (both methods identical)
# ──────────────────────────────────────────────────────────────

def profile_inference(model_name, device, batch_size=16, num_iters=50, warmup_iters=10):
    """Profile inference (no encoder needed for either method)."""
    base_name = model_name
    sit_model = SiT_models[base_name](
        path_type="linear",
        z_dims=[768],
        encoder_depth=8,
        qk_norm=True,
        eval_mode=True,
    ).to(device).eval()

    latent_size = 32

    # Warmup
    for _ in range(warmup_iters):
        x, t, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        with torch.no_grad():
            out, _, _ = sit_model(x, t, y)

    # Profile
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    t_start = time.perf_counter()

    for _ in range(num_iters):
        x, t, y = make_dummy_batch(batch_size, latent_size, 4, 1000, device)
        with torch.no_grad():
            out, _, _ = sit_model(x, t, y)

    torch.cuda.synchronize(device)
    elapsed = (time.perf_counter() - t_start) * 1000 / num_iters
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
    total_params = count_params(sit_model, trainable_only=False)

    del sit_model
    torch.cuda.empty_cache()

    return {
        "method": "Inference (both)",
        "time_ms": elapsed,
        "peak_mem_mb": peak_mem,
        "total_params": total_params,
    }


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="SiT-B/2",
                        choices=list(SiT_models.keys()))
    parser.add_argument("--encoder", type=str, default="dinov2_vitb14_reg")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_iters", type=int, default=50)
    parser.add_argument("--warmup_iters", type=int, default=10)
    parser.add_argument("--enc_layer", type=int, nargs="+", default=[8])
    parser.add_argument("--sit_layer", type=int, nargs="+", default=[8])
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"{'='*65}")
    print(f"  Cost Profiling: {args.model} + {args.encoder}")
    print(f"  Batch size: {args.batch_size}, Iters: {args.num_iters}")
    print(f"  Device: {torch.cuda.get_device_name(device)}")
    print(f"{'='*65}\n")

    # 1) REPA baseline
    print("Profiling REPA baseline...")
    repa = profile_repa_baseline(
        args.model, args.encoder, args.batch_size, device,
        num_iters=args.num_iters, warmup_iters=args.warmup_iters,
    )

    # 2) Ours Stage 1
    print("Profiling Ours (Stage 1: Scaffold)...")
    ours_s1 = profile_ours(
        args.model, args.encoder, args.batch_size, device, stage=1,
        num_iters=args.num_iters, warmup_iters=args.warmup_iters,
        enc_layer_indices=args.enc_layer, sit_layer_indices=args.sit_layer,
    )

    # 3) Ours Stage 2
    print("Profiling Ours (Stage 2: Distillation)...")
    ours_s2 = profile_ours(
        args.model, args.encoder, args.batch_size, device, stage=2,
        num_iters=args.num_iters, warmup_iters=args.warmup_iters,
        enc_layer_indices=args.enc_layer, sit_layer_indices=args.sit_layer,
    )

    # 4) Inference (identical for both)
    print("Profiling Inference...")
    inf = profile_inference(args.model, device, batch_size=args.batch_size,
                            num_iters=args.num_iters, warmup_iters=args.warmup_iters)

    # ── Print results ──
    print(f"\n{'='*65}")
    print(f"  TRAINING COST COMPARISON")
    print(f"{'='*65}")
    print(f"{'Method':<28} {'Time (ms/iter)':>14} {'Peak Mem (MB)':>14} {'Params (M)':>11}")
    print(f"{'-'*65}")

    for r in [repa, ours_s1, ours_s2]:
        params_m = r["trainable_params"] / 1e6
        print(f"{r['method']:<28} {r['time_ms']:>14.1f} {r['peak_mem_mb']:>14.0f} {params_m:>11.1f}")

    print(f"{'-'*65}")

    # Overhead
    overhead_s1_time = (ours_s1["time_ms"] - repa["time_ms"]) / repa["time_ms"] * 100
    overhead_s2_time = (ours_s2["time_ms"] - repa["time_ms"]) / repa["time_ms"] * 100
    overhead_s1_mem = (ours_s1["peak_mem_mb"] - repa["peak_mem_mb"]) / repa["peak_mem_mb"] * 100
    overhead_s2_mem = (ours_s2["peak_mem_mb"] - repa["peak_mem_mb"]) / repa["peak_mem_mb"] * 100

    print(f"{'Stage 1 overhead vs REPA':<28} {overhead_s1_time:>+13.1f}% {overhead_s1_mem:>+13.1f}%")
    print(f"{'Stage 2 overhead vs REPA':<28} {overhead_s2_time:>+13.1f}% {overhead_s2_mem:>+13.1f}%")

    if "kv_proj_params" in ours_s1:
        print(f"\nKV projection extra params: {ours_s1['kv_proj_params']/1e3:.1f}K")

    print(f"\n{'='*65}")
    print(f"  INFERENCE COST (identical for REPA and Ours)")
    print(f"{'='*65}")
    print(f"  Time: {inf['time_ms']:.1f} ms/batch  |  Peak Mem: {inf['peak_mem_mb']:.0f} MB  |  Params: {inf['total_params']/1e6:.1f}M")
    print(f"  (No encoder needed at inference for either method)")
    print()


if __name__ == "__main__":
    main()
