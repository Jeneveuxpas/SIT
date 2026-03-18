"""
generate_grid_pdf.py
--------------------
Generate 12 images and arrange them into a 2x6 grid PDF for ECCV paper.
"""

import os
import sys

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sit import SiT_models
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler


# ---------- (label, seed) pairs ----------
PAIRS = [
    (980, 0),(203, 42) , (928, 2), (970, 0), (958, 1), (84, 162),
    (849, 2),(39, 72), (51 ,2), (88, 142) , (55, 0),(933, 0),
]

CKPT = "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0400000.pt"
CFG_SCALE = 4.0
NUM_STEPS = 250
MODEL_NAME = "SiT-XL/2"
RESOLUTION = 256


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
    return samples[0]  # H x W x 3


def make_grid_pdf(images, nrows=2, ncols=6, out_path="grid_eccv.pdf"):
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)

    for idx, ax in enumerate(axes.flat):
        ax.imshow(images[idx])
        ax.set_axis_off()

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close(fig)
    print(f"Saved to {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading VAE ...")
    vae, latents_scale, latents_bias = load_vae(device)

    model, path_type = load_model(CKPT, MODEL_NAME, device, RESOLUTION)
    print(f"Using path_type={path_type}")

    images = []
    for i, (cl, seed) in enumerate(PAIRS):
        print(f"  [{i+1}/{len(PAIRS)}] label={cl} seed={seed}")
        img = generate_one(
            model=model, vae=vae,
            latents_scale=latents_scale, latents_bias=latents_bias,
            class_label=cl, seed=seed,
            cfg_scale=CFG_SCALE, num_steps=NUM_STEPS,
            path_type=path_type, resolution=RESOLUTION,
            device=device,
        )
        images.append(img)

    make_grid_pdf(images, nrows=2, ncols=6, out_path="cfg_images.pdf")


if __name__ == "__main__":
    main()
