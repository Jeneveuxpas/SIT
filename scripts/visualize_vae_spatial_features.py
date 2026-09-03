#!/usr/bin/env python3
"""Visualize what the local SD-VAE latent and mid-block feature retain spatially.

Writes one row per image: input | VAE reconstruction | x_0 PCA | mid.block_2
PCA | mid.block_2 channel energy.  PCA is fitted jointly over the requested
images so colors are comparable across rows.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.autoencoder import VAE_F8D4


def parse_indices(value: str, count: int) -> list[int]:
    if value:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return np.linspace(0, count - 1, min(8, count), dtype=np.int64).tolist()


def load_images(dataset, indices: list[int], device: torch.device) -> torch.Tensor:
    images = []
    for index in indices:
        image = dataset[index]["image"].convert("RGB")
        array = np.asarray(image, dtype=np.uint8)
        images.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
    return torch.stack(images).to(device=device, dtype=torch.float32)


def pca_rgb(features: torch.Tensor) -> torch.Tensor:
    """Map [B,C,H,W] to a globally comparable RGB PCA image in [0, 1]."""
    batch, channels, height, width = features.shape
    tokens = features.permute(0, 2, 3, 1).reshape(-1, channels).float()
    tokens = tokens - tokens.mean(dim=0, keepdim=True)
    _, _, basis = torch.pca_lowrank(tokens, q=3, center=False)
    rgb = (tokens @ basis[:, :3]).reshape(batch, height, width, 3)
    lo = rgb.amin(dim=(0, 1, 2), keepdim=True)
    hi = rgb.amax(dim=(0, 1, 2), keepdim=True)
    return ((rgb - lo) / (hi - lo).clamp_min(1e-6)).permute(0, 3, 1, 2)


def resize_uint8(tensor: torch.Tensor, size: int = 256) -> Image.Image:
    tensor = F.interpolate(tensor.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False)[0]
    array = tensor.clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def energy_rgb(features: torch.Tensor) -> torch.Tensor:
    energy = features.float().square().mean(dim=1, keepdim=True).sqrt()
    lo = energy.amin(dim=(2, 3), keepdim=True)
    hi = energy.amax(dim=(2, 3), keepdim=True)
    energy = (energy - lo) / (hi - lo).clamp_min(1e-6)
    return energy.repeat(1, 3, 1, 1)


@torch.inference_mode()
def main(args):
    device = torch.device(args.device)
    split_dir = "val" if args.split == "val" else ""
    dataset_path = Path(args.data_dir) / "imagenet-latents-images" / split_dir
    dataset = load_from_disk(str(dataset_path))
    indices = parse_indices(args.indices, len(dataset))
    if not indices or min(indices) < 0 or max(indices) >= len(dataset):
        raise ValueError(f"indices must be in [0, {len(dataset) - 1}]")

    images = load_images(dataset, indices, device)
    vae = VAE_F8D4().to(device).eval()
    checkpoint = REPO_ROOT / "pretrained_models" / "sdvae-ft-mse-f8d4.pt"
    vae.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    vae.requires_grad_(False)

    captured = {}
    def capture_mid_block2(_module, _inputs, output):
        # Forward hooks replace a module output when they return a non-None
        # value, so keep this explicitly side-effect-only.
        captured["mid"] = output.detach()

    hook = vae.encoder.mid.block_2.register_forward_hook(capture_mid_block2)
    posterior = vae.encode(images / 127.5 - 1.0)
    hook.remove()
    z0 = posterior.mean
    mid = captured.get("mid")
    if mid is None:
        raise RuntimeError("Failed to capture VAE encoder mid.block_2")
    reconstruction = vae.decode(z0).sample

    original_rgb = images / 255.0
    reconstruction_rgb = (reconstruction + 1.0) / 2.0
    z0_rgb = pca_rgb(z0)
    mid_rgb = pca_rgb(mid)
    mid_energy = energy_rgb(mid)

    labels = ["input", "VAE recon (mean z0)", "x0 PCA", "mid.block_2 PCA", "mid.block_2 energy"]
    panels = [original_rgb, reconstruction_rgb, z0_rgb, mid_rgb, mid_energy]
    cell = args.cell_size
    label_height = 24
    canvas = Image.new("RGB", (len(labels) * cell, len(indices) * (cell + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for col, label in enumerate(labels):
        draw.text((col * cell + 4, 4), label, fill="black")
    for row, index in enumerate(indices):
        y = row * (cell + label_height) + label_height
        for col, panel in enumerate(panels):
            canvas.paste(resize_uint8(panel[row], cell), (col * cell, y))
        draw.text((4, y + 4), f"index={index}", fill="white")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"Saved {output}")
    print(f"indices: {indices}")
    print(f"z0 shape: {tuple(z0.shape)}; mid.block_2 shape: {tuple(mid.shape)}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/dev/shm/data")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--indices", default="", help="Comma-separated dataset indices; default: 8 evenly spaced samples")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="visualizations/vae_spatial_features.png")
    parser.add_argument("--cell-size", type=int, default=256)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
