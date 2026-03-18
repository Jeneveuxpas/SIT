"""
generate_pyramid.py
-------------------
Generate a 2-row pyramid-grid visualization for ECCV appendix (PDF output).

Layout (same class, 12 seeds you specify):
  Row 0:  4 images at full size  (256×256)
  Row 1:  8 images at 1/2  size  (128×128)

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_pyramid.py \
        --ckpt /workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0400000.pt \
        --class-label 980 \
        --cfg-scale 4.0 \
        --num-steps 250 \
        --seeds 8 14 17 19 21 23 33 45 60 72 76 78\
        --out output/980.pdf
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sit import SiT_models
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler


# ── helpers ──────────────────────────────────────────────────────────────────

def load_vae(device):
    vae = VAE_F8D4().to(device).eval()
    base = os.path.join(os.path.dirname(__file__), "pretrained_models")
    vae_ckpt = torch.load(
        os.path.join(base, "sdvae-ft-mse-f8d4.pt"),
        map_location=device, weights_only=False,
    )
    vae.load_state_dict(vae_ckpt)
    stats = torch.load(
        os.path.join(base, "sdvae-ft-mse-f8d4-latents-stats.pt"),
        map_location=device, weights_only=False,
    )
    scale = stats["latents_scale"].to(device).view(1, -1, 1, 1)
    bias  = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return vae, scale, bias


def load_model(ckpt_path, model_name, device, resolution=256):
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_args  = ckpt.get("args", None)
    qk_norm    = getattr(ckpt_args, "qk_norm",    False) if ckpt_args else False
    path_type  = getattr(ckpt_args, "path_type", "linear") if ckpt_args else "linear"

    latent_size  = resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": qk_norm}

    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        use_cfg=True,
        eval_mode=True,
        **block_kwargs,
    ).to(device)

    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    filtered   = {k: v for k, v in state_dict.items() if "kv_proj" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  [info] {len(missing)} missing keys")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")

    model.eval()
    return model, path_type


@torch.no_grad()
def generate_batch(model, vae, latents_scale, latents_bias,
                   class_label, seeds, cfg_scale, num_steps,
                   path_type, resolution, device, batch_size=8):
    """Generate images for a list of seeds, returns list of PIL Images."""
    latent_size = resolution // 8
    images = []

    for start in range(0, len(seeds), batch_size):
        batch_seeds = seeds[start : start + batch_size]
        B = len(batch_seeds)

        # Stack noise from fixed seeds
        zs = []
        for s in batch_seeds:
            torch.manual_seed(s)
            zs.append(torch.randn(1, 4, latent_size, latent_size))
        z = torch.cat(zs, dim=0).to(device)
        y = torch.full((B,), class_label, dtype=torch.long, device=device)

        samples = euler_sampler(
            model=model, latents=z, y=y,
            num_steps=num_steps, heun=False,
            cfg_scale=cfg_scale,
            guidance_low=0.0, guidance_high=1.0,
            path_type=path_type,
        ).to(torch.float32)

        samples = vae.decode(samples / latents_scale + latents_bias).sample
        samples = (samples + 1) / 2.0
        samples = torch.clamp(255.0 * samples, 0, 255) \
                         .permute(0, 2, 3, 1).byte().cpu().numpy()

        for i in range(B):
            images.append(Image.fromarray(samples[i]))

        print(f"  generated {min(start + batch_size, len(seeds))}/{len(seeds)}", end="\r")

    print()
    return images


# ── pyramid layout ────────────────────────────────────────────────────────────

ROWS = [4, 8]                   # row 0: 4 large, row 1: 8 half-size
GAP  = 0                        # no gap between images


def build_pyramid(images, base_size=256, gap=GAP, bg_color=(255, 255, 255)):
    """
    Row 0: 4 images at base_size × base_size
    Row 1: 8 images at base_size//2 × base_size//2
    Canvas width = 4 * base_size + 3 * gap  (rows are center-aligned)
    """
    total_needed = sum(ROWS)
    if len(images) < total_needed:
        raise ValueError(f"Need exactly {total_needed} images, got {len(images)}")

    canvas_w = ROWS[0] * base_size + (ROWS[0] - 1) * gap
    canvas_h = sum(base_size // (2 ** r) for r in range(len(ROWS))) + (len(ROWS) - 1) * gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)

    idx = 0
    y_cursor = 0
    for r, n in enumerate(ROWS):
        cell_size = base_size // (2 ** r)
        row_w = n * cell_size + (n - 1) * gap
        x_start = (canvas_w - row_w) // 2

        for c in range(n):
            img = images[idx].resize((cell_size, cell_size), Image.LANCZOS)
            x = x_start + c * (cell_size + gap)
            canvas.paste(img, (x, y_cursor))
            idx += 1

        y_cursor += cell_size + gap

    return canvas


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate 2-row pyramid-grid PDF figure.")
    parser.add_argument("--ckpt",        type=str,   required=True)
    parser.add_argument("--class-label", type=int,   required=True)
    parser.add_argument("--cfg-scale",   type=float, default=4.0)
    parser.add_argument("--num-steps",   type=int,   default=250)
    parser.add_argument("--seeds",       type=int,   nargs="+", required=True,
                        metavar="S",
                        help="Exactly 12 seed values (first 4 → large row, next 8 → small row)")
    parser.add_argument("--path-type",   type=str,   default=None,
                        choices=["linear", "cosine"])
    parser.add_argument("--resolution",  type=int,   default=256, choices=[256, 512])
    parser.add_argument("--model",       type=str,   default="SiT-XL/2",
                        choices=list(SiT_models.keys()))
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--gap",         type=int,   default=GAP,
                        help="Pixel gap between images in grid")
    parser.add_argument("--bg-color",    type=int,   nargs=3, default=[255, 255, 255],
                        metavar=("R", "G", "B"))
    parser.add_argument("--out",         type=str,   default="pyramid.pdf")
    args = parser.parse_args()

    total_needed = sum(ROWS)  # 12
    if len(args.seeds) != total_needed:
        parser.error(f"--seeds requires exactly {total_needed} values, got {len(args.seeds)}")

    seeds = args.seeds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading VAE ...")
    vae, latents_scale, latents_bias = load_vae(device)

    model, ckpt_path_type = load_model(args.ckpt, args.model, device, args.resolution)
    path_type = args.path_type if args.path_type is not None else ckpt_path_type
    print(f"Using path_type={path_type}")

    print(f"Generating {len(seeds)} images for class {args.class_label} ...")
    images = generate_batch(
        model=model, vae=vae,
        latents_scale=latents_scale, latents_bias=latents_bias,
        class_label=args.class_label,
        seeds=seeds,
        cfg_scale=args.cfg_scale,
        num_steps=args.num_steps,
        path_type=path_type,
        resolution=args.resolution,
        device=device,
        batch_size=args.batch_size,
    )

    print("Composing pyramid ...")
    canvas = build_pyramid(
        images,
        base_size=args.resolution,
        gap=args.gap,
        bg_color=tuple(args.bg_color),
    )

    # Upsample 2× for print-quality PDF (effective ~427 DPI at ECCV text width)
    canvas = canvas.resize((canvas.width * 2, canvas.height * 2), Image.LANCZOS)

    # Save as PDF (300 DPI metadata)
    out = args.out if args.out.lower().endswith(".pdf") else args.out + ".pdf"
    canvas.save(out, "PDF", resolution=300)
    print(f"Saved → {out}  ({canvas.width}×{canvas.height} px)")


if __name__ == "__main__":
    main()
