#!/usr/bin/env python3
"""Compare precomputed training VAE moments with the x_0 oracle encoder."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.autoencoder import VAE_F8D4


def load_batch(image_dataset, latent_dataset, indices, device):
    images = []
    moments = []
    for index in indices:
        image = np.asarray(image_dataset[int(index)]["image"].convert("RGB"), dtype=np.uint8)
        images.append(torch.from_numpy(image.copy()).permute(2, 0, 1))

        stored = torch.as_tensor(latent_dataset[int(index)]["data"])
        # Some HF datasets retain the encoder's singleton batch dimension.
        if stored.ndim == 4 and stored.shape[0] == 1:
            stored = stored.squeeze(0)
        moments.append(stored)

    return (
        torch.stack(images).to(device=device, dtype=torch.float32),
        torch.stack(moments).to(device=device, dtype=torch.float32),
    )


@torch.inference_mode()
def main(args):
    device = torch.device(args.device)
    split_dir = "val" if args.split == "val" else ""
    image_path = Path(args.data_dir) / "imagenet-latents-images" / split_dir
    latent_path = Path(args.data_dir) / "imagenet-latents-sdvae-ft-mse-f8d4" / split_dir
    image_dataset = load_from_disk(str(image_path))
    latent_dataset = load_from_disk(str(latent_path))
    if len(image_dataset) != len(latent_dataset):
        raise ValueError(
            f"Image/latent counts differ: {len(image_dataset)} vs {len(latent_dataset)}"
        )

    count = min(args.count, len(image_dataset))
    indices = np.linspace(0, len(image_dataset) - 1, count, dtype=np.int64)
    images, stored = load_batch(image_dataset, latent_dataset, indices, device)
    if stored.ndim != 4 or stored.shape[1] != 8:
        raise ValueError(f"Expected stored moments [B,8,H,W], got {tuple(stored.shape)}")

    checkpoint_path = REPO_ROOT / "pretrained_models" / "sdvae-ft-mse-f8d4.pt"
    vae = VAE_F8D4().to(device).eval()
    vae.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)
    )
    vae.requires_grad_(False)

    posterior = vae.encode(images / 127.5 - 1.0)
    online = torch.cat((posterior.mean, posterior.std), dim=1).float()
    if online.shape != stored.shape:
        raise ValueError(
            f"Online/stored shapes differ: {tuple(online.shape)} vs {tuple(stored.shape)}"
        )

    delta = online - stored
    flat_online = online.flatten(1)
    flat_stored = stored.flatten(1)
    cosine = F.cosine_similarity(flat_online, flat_stored, dim=1)
    stored_rms = stored.square().mean().sqrt()
    relative_rmse = delta.square().mean().sqrt() / stored_rms.clamp_min(1e-12)

    print(f"samples: {count}")
    print(f"shape: {tuple(online.shape)}")
    print(f"mean_abs_diff: {delta.abs().mean().item():.8g}")
    print(f"max_abs_diff: {delta.abs().max().item():.8g}")
    print(f"relative_rmse: {relative_rmse.item():.8g}")
    print(f"cosine_mean: {cosine.mean().item():.8g}")
    print(f"cosine_min: {cosine.min().item():.8g}")

    if relative_rmse < args.relative_rmse_tol and cosine.min() > args.cosine_tol:
        print("PASS: oracle VAE moments match the precomputed training moments.")
    else:
        print("FAIL: oracle VAE moments do not match the precomputed training moments.")
        raise SystemExit(2)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/dev/shm/data")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--relative-rmse-tol", type=float, default=1e-4)
    parser.add_argument("--cosine-tol", type=float, default=0.999999)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
