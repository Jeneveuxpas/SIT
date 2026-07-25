#!/usr/bin/env python3
"""Generate a reference-conditioned oracle sample batch for FID evaluation."""

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from samplers import euler_maruyama_sampler, euler_sampler
from scripts.sample_hidden_replacement_oracle import (
    FixedHiddenScaffold,
    FixedKVScaffold,
    decode_latents,
    final_features_to_kv_memory,
    load_local_vae,
    load_oracle_model,
    make_seeded_latents,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def center_crop_arr(image: Image.Image, image_size: int) -> np.ndarray:
    """ADM center crop, matching dataset.py without importing HF datasets."""
    image = image.convert("RGB")
    while min(image.size) >= 2 * image_size:
        image = image.resize(
            tuple(size // 2 for size in image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(image.size)
    image = image.resize(
        tuple(round(size * scale) for size in image.size),
        resample=Image.Resampling.BICUBIC,
    )
    array = np.asarray(image, dtype=np.uint8)
    top = (array.shape[0] - image_size) // 2
    left = (array.shape[1] - image_size) // 2
    return array[top : top + image_size, left : left + image_size]


def discover_references(root: Path):
    # The training server stores validation images as a Hugging Face Dataset.
    if (root / "dataset_info.json").exists() or (root / "state.json").exists():
        from datasets import load_from_disk

        dataset = load_from_disk(str(root))
        if "image" not in dataset.column_names or "label" not in dataset.column_names:
            raise ValueError(
                f"HF reference dataset at {root} must contain image and label columns; "
                f"found {dataset.column_names}"
            )
        print(f"Loaded {len(dataset)} references from HF dataset: {root}")
        return dataset

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No ImageNet class directories found under {root}")
    samples = []
    for label, class_dir in enumerate(class_dirs):
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, label))
    if not samples:
        raise FileNotFoundError(f"No reference images found under {root}")
    return samples


def load_reference_batch(samples, resolution: int) -> tuple[torch.Tensor, torch.Tensor]:
    images, labels = [], []
    for sample in samples:
        if isinstance(sample, dict):
            image, label = sample["image"], int(sample["label"])
        else:
            path, label = sample
            image = Image.open(path)
        array = center_crop_arr(image, resolution)
        images.append(torch.from_numpy(array.copy()).permute(2, 0, 1))
        labels.append(label)
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def create_npz(sample_dir: Path, output_path: Path, count: int):
    images = []
    for index in tqdm(range(count), desc="Packing oracle samples"):
        images.append(
            np.asarray(Image.open(sample_dir / f"{index:06d}.png"), dtype=np.uint8)
        )
    array = np.stack(images)
    np.savez(output_path, arr_0=array)
    print(f"Saved oracle sample batch: {output_path} {array.shape}")


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle FID generation requires CUDA")
    if args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be >= 1")

    device = torch.device("cuda")
    dtype = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[args.inference_dtype]
    torch.backends.cuda.matmul.allow_tf32 = args.tf32

    model, encoder, extractor, metadata = load_oracle_model(
        args.checkpoint, device, dtype
    )
    references = discover_references(Path(args.reference_dir))
    count = min(args.num_fid_samples, len(references))
    if count < args.num_fid_samples:
        raise ValueError(
            f"Requested {args.num_fid_samples} references, found only {len(references)}"
        )

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    expected_tokens = (metadata["latent_size"] // 2) ** 2
    vae, scale, bias = load_local_vae(device, args.vae)

    total_batches = math.ceil(count / args.batch_size)
    for batch_index in tqdm(range(total_batches), desc="Oracle sampling"):
        start = batch_index * args.batch_size
        end = min(start + args.batch_size, count)
        batch_refs = [references[index] for index in range(start, end)]
        raw_images, labels = load_reference_batch(batch_refs, metadata["resolution"])
        raw_images = raw_images.to(device=device, dtype=torch.float32)
        labels = labels.to(device)

        encoded_input = encoder.preprocess(raw_images)
        if extractor is not None:
            extractor.reset_cache()
            extractor._batch_size = encoded_input.shape[0]
        encoder_outputs = encoder.forward_features(encoded_input)
        clean_feature = encoder_outputs["x_norm_patchtokens"]

        if metadata["interface"] in ("hidden", "residual"):
            feature = clean_feature.to(device=device, dtype=dtype)
            if feature.shape[1] != expected_tokens:
                raise ValueError(
                    f"Encoder returned {feature.shape[1]} tokens; expected {expected_tokens}"
                )
            oracle_model = FixedHiddenScaffold(model, feature).eval()
        else:
            if metadata["feature_source"] == "final_feature":
                memory = final_features_to_kv_memory(
                    clean_feature,
                    num_heads=metadata["encoder_heads"],
                    target_num_patches=expected_tokens,
                )
                encoder_kv = [
                    (None, memory, memory) for _ in metadata["enc_layers"]
                ]
            else:
                encoder_kv = extractor.get_captured_kv_list()
            encoder_kv = [
                tuple(
                    None
                    if component is None
                    else component.to(device=device, dtype=dtype)
                    for component in layer_kv
                )
                for layer_kv in encoder_kv
            ]
            oracle_model = FixedKVScaffold(model, encoder_kv).eval()

        seeds = [args.global_seed + index for index in range(start, end)]
        latents = make_seeded_latents(
            seeds, metadata["latent_size"], device=device, dtype=dtype
        )
        sampling_kwargs = dict(
            model=oracle_model,
            latents=latents,
            y=labels,
            num_steps=args.num_steps,
            heun=args.heun,
            cfg_scale=args.cfg_scale,
            guidance_low=args.guidance_low,
            guidance_high=args.guidance_high,
            path_type=metadata["path_type"],
        )
        if args.mode == "sde":
            torch.manual_seed(args.global_seed + start)
            generated_latents = euler_maruyama_sampler(**sampling_kwargs)
        else:
            generated_latents = euler_sampler(**sampling_kwargs)

        generated = decode_latents(generated_latents, vae, scale, bias)
        for offset, image in enumerate(generated):
            image.save(image_dir / f"{start + offset:06d}.png")

        del oracle_model, generated_latents, generated

    if extractor is not None:
        extractor.remove_hooks()
    create_npz(image_dir, Path(args.output_npz), count)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--mode", choices=["ode", "sde"], default="sde")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--vae", choices=["mse", "ema"], default="ema")
    parser.add_argument(
        "--inference-dtype",
        choices=["fp32", "fp16", "bf16"],
        default="fp32",
    )
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
