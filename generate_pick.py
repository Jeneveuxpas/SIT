"""
generate_pick.py
----------------
Generate images from a single checkpoint for cherry-picking.

Given a checkpoint, cfg scale, class labels, and seeds, generates all
(label x seed) combinations using ODE sampling and saves each image
individually for easy browsing.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_pick.py \
        --ckpt /path/to/checkpoint.pt \
        --cfg-scale 1.8 \
        --class-labels 207 88 360 270 \
        --seeds 0 1 2 42 \
        --num-steps 250 \
        --out-dir pick_output
"""

import argparse
import itertools
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sit import SiT_models
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler


IMAGENET_NAMES = {
    0: "tench", 1: "goldfish", 2: "great_white_shark", 7: "hen",
    88: "macaw", 89: "sulphur_crested_cockatoo", 130: "flamingo",
    200: "Tibetan_terrier", 207: "golden_retriever", 270: "white_wolf",
    279: "Arctic_fox", 281: "tabby_cat", 291: "lion",
    325: "fire_truck", 340: "zebra", 360: "otter",
    388: "giant_panda", 400: "hot_air_balloon", 417: "balloon",
    555: "fire_engine", 717: "pickup_truck", 724: "sports_car",
    849: "cheeseburger", 933: "cheeseburger", 985: "daisy",
}


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
    bias = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return vae, scale, bias


def load_model(ckpt_path, model_name, device, resolution=256):
    print(f"Loading {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_args = ckpt.get("args", None)
    qk_norm = False
    path_type = "linear"
    if ckpt_args is not None:
        if hasattr(ckpt_args, "qk_norm"):
            qk_norm = ckpt_args.qk_norm
        if hasattr(ckpt_args, "path_type"):
            path_type = ckpt_args.path_type

    latent_size = resolution // 8
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
    filtered = {k: v for k, v in state_dict.items() if "kv_proj" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"  [info] {len(missing)} missing keys")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys")

    model.eval()
    return model, path_type


@torch.no_grad()
def generate_one(model, vae, latents_scale, latents_bias,
                 class_label, seed, cfg_scale, num_steps,
                 path_type, resolution, device):
    latent_size = resolution // 8
    torch.manual_seed(seed)

    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    y = torch.full((1,), class_label, dtype=torch.long, device=device)

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
    return Image.fromarray(samples[0])


def main():
    parser = argparse.ArgumentParser(description="Generate images for cherry-picking.")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--cfg-scale", type=float, default=1.8)
    parser.add_argument("--class-labels", type=int, nargs="+", required=True,
                        metavar="CLS")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        metavar="S")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--path-type", type=str, default=None,
                        choices=["linear", "cosine"],
                        help="Override path_type (auto-detected from ckpt if not set)")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()))
    parser.add_argument("--out-dir", type=str, default="pick_output")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Load VAE
    print("Loading VAE ...")
    vae, latents_scale, latents_bias = load_vae(device)

    # Load model
    model, ckpt_path_type = load_model(args.ckpt, args.model, device, args.resolution)
    path_type = args.path_type if args.path_type is not None else ckpt_path_type
    print(f"Using path_type={path_type}")

    # Generate all combinations
    combos = list(itertools.product(args.class_labels, args.seeds))
    total = len(combos)
    print(f"\nGenerating {total} images "
          f"({len(args.class_labels)} labels x {len(args.seeds)} seeds) ...")

    for i, (cl, seed) in enumerate(combos):
        cls_name = IMAGENET_NAMES.get(cl, f"cls{cl}")
        print(f"  [{i+1}/{total}] label={cl}({cls_name}) seed={seed}", end="\r")

        img = generate_one(
            model=model, vae=vae,
            latents_scale=latents_scale, latents_bias=latents_bias,
            class_label=cl, seed=seed,
            cfg_scale=args.cfg_scale, num_steps=args.num_steps,
            path_type=path_type, resolution=args.resolution,
            device=device,
        )

        fname = f"cls{cl:04d}_{cls_name}_seed{seed:04d}.png"
        img.save(os.path.join(args.out_dir, fname))

    print(f"\nDone! {total} images saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
