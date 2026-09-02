#!/usr/bin/env python3
"""Oracle clean-image diagnostic for a Stage-1 hidden or K/V checkpoint.

This is not standard generation: the reference clean image is encoded once by
the frozen visual encoder, and its patch features replace the selected SiT
hidden state at every sampling step. The script is intended to visualize what
the downstream SiT blocks learned to decode during the scaffold stage.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/sample_hidden_replacement_oracle.py \
        --checkpoint exps/attnscaf-hidden-replacement-layer8-no-consistency-100k/checkpoints/0030000.pt \
        --reference-image /path/to/reference.JPEG \
        --class-label 207 \
        --seeds 0 1 2 3 \
        --output-dir oracle_hidden_layer8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.encoder_adapter import EncoderKVExtractor, VAEEncoderKVExtractor
from models.autoencoder import VAE_F8D4
from models.sit_encoder import SiT_EncoderKV_models
from samplers import euler_maruyama_sampler, euler_sampler
from vision_encoder import load_encoders


def get_arg(saved_args, name: str, default=None):
    """Read a checkpoint argument from either a Namespace or a dictionary."""
    if saved_args is None:
        return default
    if isinstance(saved_args, dict):
        return saved_args.get(name, default)
    return getattr(saved_args, name, default)


def parse_1based_indices(value) -> list[int]:
    if isinstance(value, str):
        return [int(item.strip()) - 1 for item in value.split(",")]
    if isinstance(value, Iterable):
        return [int(item) - 1 for item in value]
    raise ValueError(f"Cannot parse layer indices from {value!r}")


def final_features_to_kv_memory(
    features: torch.Tensor,
    num_heads: int,
    target_num_patches: int,
) -> torch.Tensor:
    """Match train.py's layout conversion for final-feature K/V scaffolds."""
    if features.ndim != 3:
        raise ValueError(
            f"Expected final features with shape (B, N, C), got {tuple(features.shape)}"
        )
    batch, num_tokens, channels = features.shape
    if channels % num_heads != 0:
        raise ValueError(
            f"Feature dimension {channels} is not divisible by {num_heads} heads"
        )
    if num_tokens != target_num_patches:
        source_hw = int(round(num_tokens ** 0.5))
        target_hw = int(round(target_num_patches ** 0.5))
        if source_hw * source_hw != num_tokens or target_hw * target_hw != target_num_patches:
            raise ValueError(
                f"Cannot resize final features from {num_tokens} to "
                f"{target_num_patches} non-square tokens"
            )
        original_dtype = features.dtype
        features_2d = features.transpose(1, 2).reshape(
            batch, channels, source_hw, source_hw
        )
        features_2d = torch.nn.functional.interpolate(
            features_2d.float(),
            size=(target_hw, target_hw),
            mode="bilinear",
            align_corners=False,
        )
        features = features_2d.flatten(2).transpose(1, 2).to(original_dtype)
        num_tokens = target_num_patches
    return features.reshape(
        batch, num_tokens, num_heads, channels // num_heads
    ).transpose(1, 2).detach()


def select_oracle_feature(
    clean_feature: torch.Tensor,
    extractor: EncoderKVExtractor | None,
    feature_source: str,
) -> torch.Tensor:
    """Select the same encoder feature source used to train a residual scaffold."""
    if feature_source == "repa":
        return clean_feature
    if extractor is None:
        raise RuntimeError(
            f"scaffold_feature_source={feature_source!r} requires an encoder extractor"
        )
    if feature_source == "attn_input":
        features = extractor.get_captured_feat_list()
    elif feature_source == "attn_output":
        features = extractor.get_captured_attn_output_list()
    else:
        raise ValueError(
            "Residual oracle inference supports repa, attn_input, or attn_output "
            f"feature sources, got {feature_source!r}"
        )
    if len(features) != 1:
        raise ValueError(
            "The fixed residual oracle currently expects exactly one encoder/SiT "
            f"layer pair, found {len(features)}"
        )
    return features[0]


def load_reference_image(path: str, resolution: int) -> tuple[Image.Image, torch.Tensor]:
    """Center-crop the reference and return both PIL and raw [0,255] BCHW."""
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(
        image,
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )
    array = np.asarray(image, dtype=np.float32)
    raw = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return image, raw


def load_local_vae(device: torch.device, vae_name: str):
    # Reuse the standard evaluation path exactly:
    #   1. load the requested decoder (e.g. sd-vae-ft-ema);
    #   2. if its latent-statistics file is absent, use the MSE statistics;
    #   3. only use a converted local decoder when the HF decoder is unavailable.
    from generate import load_vae

    return load_vae(device=device, vae_name=vae_name)


def load_latent_oracle_vae(device: torch.device):
    """Load the *training* MSE VAE encoder and latent statistics for x_0.

    This intentionally does not use ``--vae``: that option chooses the decoder
    used to score generated images, while latent-source training always uses
    the local sd-vae-ft-mse encoder and its corresponding latent statistics.
    """
    checkpoint_path = REPO_ROOT / "pretrained_models" / "sdvae-ft-mse-f8d4.pt"
    stats_path = REPO_ROOT / "pretrained_models" / "sdvae-ft-mse-f8d4-latents-stats.pt"
    vae = VAE_F8D4().to(device).eval()
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vae.load_state_dict(state_dict)
    vae.requires_grad_(False)
    stats = torch.load(stats_path, map_location=device, weights_only=False)
    scale = stats["latents_scale"].to(device).view(1, -1, 1, 1)
    bias = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return vae, scale, bias


@torch.inference_mode()
def raw_images_to_latent_kv(
    raw_images: torch.Tensor,
    vae: torch.nn.Module,
    scale: torch.Tensor,
    bias: torch.Tensor,
    patch_size: int = 2,
) -> torch.Tensor:
    """Match train.py's sampled, normalized clean-latent K/V source."""
    posterior = vae.encode(raw_images.float() / 127.5 - 1.0)
    # ``posterior.parameters`` is (mean, logvar).  The precomputed dataset
    # consumed by train.py stores (mean, std), so using the raw second half
    # here incorrectly treated logvar as a standard deviation.
    x0 = (
        posterior.mean
        + posterior.std * torch.randn_like(posterior.mean)
        - bias
    ) * scale
    batch, channels, height, width = x0.shape
    if height % patch_size or width % patch_size:
        raise ValueError(f"Latent grid {(height, width)} is not divisible by {patch_size}")
    tokens = x0.reshape(
        batch, channels, height // patch_size, patch_size, width // patch_size, patch_size
    )
    tokens = tokens.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, (height // patch_size) * (width // patch_size), channels * patch_size * patch_size
    )
    return tokens.unsqueeze(1).contiguous()


def load_oracle_model(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved_args = checkpoint.get("args")
    if saved_args is None:
        raise ValueError("Checkpoint does not contain saved training args")

    model_name = get_arg(saved_args, "model", "SiT-XL/2-EncoderKV")
    if model_name not in SiT_EncoderKV_models:
        raise ValueError(
            f"Expected an EncoderKV checkpoint, got model={model_name!r}"
        )

    interface = get_arg(saved_args, "scaffold_interface", "kv")
    feature_source = get_arg(saved_args, "scaffold_feature_source", "attn_input")
    if interface not in ("hidden", "residual", "kv"):
        raise ValueError(
            f"This diagnostic supports hidden, residual, or kv checkpoints, got {interface!r}"
        )
    if interface == "hidden" and feature_source != "repa":
        raise ValueError(
            "Hidden oracle inference requires scaffold_feature_source='repa'"
        )
    if interface == "residual" and feature_source not in (
        "repa",
        "attn_input",
        "attn_output",
    ):
        raise ValueError(
            "Residual oracle inference supports repa, attn_input, or attn_output "
            f"feature sources, got {feature_source!r}"
        )
    if interface == "kv" and feature_source not in (
        "attn_input", "final_feature", "latent", "vae_attn", "vae_mid_block2"
    ):
        raise ValueError(
            "K/V oracle inference supports attn_input, final_feature, latent, or "
            "vae_attn, or vae_mid_block2 sources, "
            f"got {feature_source!r}"
        )

    resolution = int(get_arg(saved_args, "resolution", 256))
    latent_size = resolution // 8
    enc_type = get_arg(saved_args, "enc_type", "dinov2-b")
    uses_latent_source = interface == "kv" and feature_source == "latent"
    uses_vae_attn_source = interface == "kv" and feature_source == "vae_attn"
    uses_vae_mid_feature_source = (
        interface == "kv" and feature_source == "vae_mid_block2"
    )
    encoder = None
    encoder_dim = 0
    if not (uses_latent_source or uses_vae_attn_source or uses_vae_mid_feature_source):
        print(f"Loading visual encoder: {enc_type}")
        encoder = load_encoders(enc_type, device, resolution)[0]
        encoder.eval()
        encoder_dim = int(encoder.embed_dim)

    enc_layer_indices = parse_1based_indices(
        get_arg(saved_args, "enc_layer_indices", "12")
    )
    sit_layer_indices = parse_1based_indices(
        get_arg(saved_args, "sit_layer_indices", "8")
    )
    if len(enc_layer_indices) != len(sit_layer_indices):
        raise ValueError("Encoder and SiT layer-index lists have different lengths")

    extractor = None
    needs_extractor = (
        interface == "kv" and not (
            uses_latent_source or uses_vae_attn_source or uses_vae_mid_feature_source
        )
    ) or (
        interface == "residual" and feature_source in ("attn_input", "attn_output")
    )
    if needs_extractor:
        extractor = EncoderKVExtractor(encoder.model, enc_layer_indices).eval()
        extractor._target_num_patches = (latent_size // 2) ** 2
        if interface == "kv":
            detected_dim = extractor.get_layer_dim(enc_layer_indices[0])
        else:
            detected_dim = extractor.get_layer_input_dim(enc_layer_indices[0])
        detected_heads = extractor.get_layer_heads(enc_layer_indices[0])
        encoder_kv_dim = detected_dim or encoder_dim
        encoder_heads = detected_heads or max(1, encoder_kv_dim // 64)
    elif uses_latent_source:
        encoder_kv_dim = 4 * 2 * 2
        encoder_heads = 1
    elif uses_vae_attn_source:
        vae_grid = resolution // 8
        sit_grid = latent_size // 2
        if vae_grid % sit_grid:
            raise ValueError(
                f"VAE grid {vae_grid} is not divisible by SiT grid {sit_grid}"
            )
        encoder_kv_dim = 512 * (vae_grid // sit_grid) ** 2
        encoder_heads = 1
    elif uses_vae_mid_feature_source:
        encoder_kv_dim = 512
        encoder_heads = 1
    else:
        encoder_kv_dim = encoder_dim
        encoder_heads = 12

    model = SiT_EncoderKV_models[model_name](
        path_type=get_arg(saved_args, "path_type", "linear"),
        input_size=latent_size,
        in_channels=4,
        num_classes=int(get_arg(saved_args, "num_classes", 1000)),
        use_cfg=float(get_arg(saved_args, "cfg_prob", 0.1)) > 0,
        z_dims=[encoder_dim],
        encoder_depth=int(get_arg(saved_args, "encoder_depth", 10)),
        eval_mode=True,
        projection_layer_type=get_arg(saved_args, "projection_layer_type", "mlp"),
        proj_kwargs_kernel_size=int(
            get_arg(saved_args, "proj_kwargs_kernel_size", 3)
        ),
        enc_layer_indices=enc_layer_indices,
        sit_layer_indices=sit_layer_indices,
        enc_dim=encoder_kv_dim,
        enc_heads=encoder_heads,
        kv_proj_type=get_arg(saved_args, "kv_proj_type", "linear"),
        kv_proj_hidden_dim=get_arg(saved_args, "kv_proj_hidden_dim", None),
        kv_proj_kernel_size=int(get_arg(saved_args, "kv_proj_kernel_size", 1)),
        kv_proj_stride=int(get_arg(saved_args, "kv_proj_stride", 1)),
        kv_norm_type=get_arg(saved_args, "kv_norm_type", "none"),
        kv_post_norm_type=get_arg(saved_args, "kv_post_norm_type", "none"),
        kv_zscore_alpha=float(get_arg(saved_args, "kv_zscore_alpha", 1.0)),
        kv_replace_mode=get_arg(saved_args, "kv_replace_mode", "kv"),
        kv_memory_mode=get_arg(saved_args, "kv_memory_mode", "replace"),
        scaffold_interface=interface,
        fused_attn=bool(get_arg(saved_args, "fused_attn", True)),
        qk_norm=bool(get_arg(saved_args, "qk_norm", False)),
    )

    state_dict = checkpoint.get("ema", checkpoint.get("model"))
    if state_dict is None:
        raise ValueError("Checkpoint contains neither 'ema' nor 'model' weights")
    state_dict = {
        key: value for key, value in state_dict.items() if "projectors" not in key
    }
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        raise RuntimeError(f"Missing model weights: {missing[:20]}")
    if unexpected:
        print(f"Ignoring {len(unexpected)} unexpected checkpoint keys: {unexpected[:5]}")

    model = model.to(device=device, dtype=dtype).eval()
    metadata = {
        "model_name": model_name,
        "resolution": resolution,
        "latent_size": latent_size,
        "enc_type": enc_type,
        "interface": interface,
        "feature_source": feature_source,
        "uses_latent_source": uses_latent_source,
        "uses_vae_attn_source": uses_vae_attn_source,
        "uses_vae_mid_feature_source": uses_vae_mid_feature_source,
        "vae_mid_norm_out_silu": bool(
            get_arg(saved_args, "vae_mid_norm_out_silu", False)
        ),
        "kv_memory_mode": get_arg(saved_args, "kv_memory_mode", "replace"),
        "enc_layers": [index + 1 for index in enc_layer_indices],
        "sit_layers": [index + 1 for index in sit_layer_indices],
        "encoder_heads": encoder_heads,
        "path_type": get_arg(saved_args, "path_type", "linear"),
        "checkpoint_step": int(checkpoint.get("steps", -1)),
    }
    return model, encoder, extractor, metadata


class FixedHiddenScaffold(torch.nn.Module):
    """Bind one clean-image feature to every call made by a sampler."""

    def __init__(self, model: torch.nn.Module, clean_feature: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("clean_feature", clean_feature, persistent=False)

    @property
    def in_channels(self):
        return self.model.in_channels

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        feature = self.clean_feature
        if feature.shape[0] == 1:
            feature = feature.expand(x.shape[0], -1, -1)
        elif x.shape[0] % feature.shape[0] == 0:
            feature = feature.repeat(x.shape[0] // feature.shape[0], 1, 1)
        elif feature.shape[0] != x.shape[0]:
            raise ValueError(
                f"Feature batch {feature.shape[0]} does not match sampler batch {x.shape[0]}"
            )

        return self.model(
            x,
            t,
            y,
            enc_feat_list=[feature],
            stage=1,
            transition_alpha=torch.zeros((), device=x.device, dtype=x.dtype),
            transition_active=False,
            enable_scaffold_in_eval=True,
        )


class FixedKVScaffold(torch.nn.Module):
    """Bind clean-image encoder K/V to every call made by a sampler."""

    def __init__(self, model: torch.nn.Module, encoder_kv: list[tuple]):
        super().__init__()
        self.model = model
        self.encoder_kv = encoder_kv

    @property
    def in_channels(self):
        return self.model.in_channels

    @staticmethod
    def _expand_component(component, batch_size: int):
        if component is None:
            return None
        if component.shape[0] == 1:
            return component.expand(batch_size, -1, -1, -1)
        if batch_size % component.shape[0] == 0:
            return component.repeat(batch_size // component.shape[0], 1, 1, 1)
        if component.shape[0] != batch_size:
            raise ValueError(
                f"K/V batch {component.shape[0]} does not match sampler batch {batch_size}"
            )
        return component

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor):
        expanded_kv = [
            tuple(self._expand_component(component, x.shape[0]) for component in layer_kv)
            for layer_kv in self.encoder_kv
        ]
        return self.model(
            x,
            t,
            y,
            enc_kv_list=expanded_kv,
            stage=1,
            transition_alpha=torch.zeros((), device=x.device, dtype=x.dtype),
            transition_active=False,
            enable_scaffold_in_eval=True,
        )


def make_seeded_latents(
    seeds: list[int], latent_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    latents = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(seed)
        latents.append(
            torch.randn(
                1,
                4,
                latent_size,
                latent_size,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
    return torch.cat(latents, dim=0)


def decode_latents(
    latents: torch.Tensor,
    vae: torch.nn.Module,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> list[Image.Image]:
    decoded = vae.decode(latents.float() / scale + bias).sample
    decoded = (decoded + 1.0) / 2.0
    decoded = decoded.clamp(0, 1).mul(255).permute(0, 2, 3, 1)
    arrays = decoded.to(device="cpu", dtype=torch.uint8).numpy()
    return [Image.fromarray(array) for array in arrays]


def save_montage(
    reference: Image.Image,
    generated: list[Image.Image],
    seeds: list[int],
    output_path: Path,
):
    images = [reference] + generated
    labels = ["reference"] + [f"seed {seed}" for seed in seeds]
    width = sum(image.width for image in images)
    label_height = 24
    canvas = Image.new("RGB", (width, reference.height + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    x_offset = 0
    for image, label in zip(images, labels):
        canvas.paste(image, (x_offset, label_height))
        draw.text((x_offset + 5, 5), label, fill="black")
        x_offset += image.width
    canvas.save(output_path)


@torch.inference_mode()
def main(args):
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")

    device = torch.device(args.device)
    dtype_by_name = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_by_name[args.dtype]
    if device.type == "cpu" and dtype != torch.float32:
        raise ValueError("Use --dtype fp32 for CPU execution")
    if not args.seeds:
        raise ValueError("At least one seed is required")
    if not 0 <= args.class_label < 1000:
        raise ValueError("--class-label must be an ImageNet class index in [0, 999]")
    if args.cfg_scale < 1.0:
        raise ValueError("--cfg-scale must be at least 1.0")

    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, encoder, extractor, metadata = load_oracle_model(
        args.checkpoint, device, dtype
    )
    if metadata["checkpoint_step"] != 30000:
        print(
            f"[warning] checkpoint step is {metadata['checkpoint_step']}, not 30000; "
            "the diagnostic is intended for the end of Stage 1"
        )

    reference, raw_reference = load_reference_image(
        args.reference_image, metadata["resolution"]
    )
    raw_reference = raw_reference.to(device)
    encoded_input = encoder.preprocess(raw_reference)
    if extractor is not None:
        extractor.reset_cache()
        extractor._batch_size = encoded_input.shape[0]
    encoder_outputs = encoder.forward_features(encoded_input)
    clean_feature = encoder_outputs["x_norm_patchtokens"]
    if clean_feature is None:
        raise RuntimeError("Visual encoder did not return normalized patch tokens")
    expected_tokens = (metadata["latent_size"] // 2) ** 2

    if metadata["interface"] in ("hidden", "residual"):
        oracle_feature = select_oracle_feature(
            clean_feature,
            extractor,
            metadata["feature_source"],
        ).to(device=device, dtype=dtype)
        if oracle_feature.shape[1] != expected_tokens:
            raise ValueError(
                f"Oracle feature has {oracle_feature.shape[1]} tokens, but SiT expects "
                f"{expected_tokens}; check the reference resolution and encoder"
            )
        oracle_payload = oracle_feature
        oracle_model = FixedHiddenScaffold(model, oracle_payload).eval()
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
                None if component is None else component.to(device=device, dtype=dtype)
                for component in layer_kv
            )
            for layer_kv in encoder_kv
        ]
        for _, keys, values in encoder_kv:
            if keys.shape[2] != expected_tokens or values.shape[2] != expected_tokens:
                raise ValueError(
                    f"Encoder K/V token count does not match SiT: "
                    f"K={keys.shape[2]}, V={values.shape[2]}, expected={expected_tokens}"
                )
        oracle_payload = encoder_kv
        oracle_model = FixedKVScaffold(model, oracle_payload).eval()

    # The encoder is no longer needed: the same feature/K/V is reused every step.
    if extractor is not None:
        extractor.remove_hooks()
    del extractor, encoder, encoder_outputs, encoded_input, raw_reference, clean_feature
    if device.type == "cuda":
        torch.cuda.empty_cache()
    latents = make_seeded_latents(
        args.seeds, metadata["latent_size"], device, dtype
    )
    labels = torch.full(
        (len(args.seeds),),
        args.class_label,
        device=device,
        dtype=torch.long,
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
    if args.mode == "ode":
        samples = euler_sampler(**sampling_kwargs)
    else:
        # Make the stochastic increments repeatable for a fixed invocation.
        torch.manual_seed(args.sde_seed)
        samples = euler_maruyama_sampler(**sampling_kwargs)

    # Free the large SiT before loading the VAE used only for decoding.
    del oracle_model, model, oracle_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    vae, scale, bias = load_local_vae(device, args.vae)
    generated = decode_latents(samples, vae, scale, bias)

    reference.save(output_dir / "reference.png")
    for image, seed in zip(generated, args.seeds):
        image.save(output_dir / f"oracle_seed{seed:06d}.png")
    save_montage(reference, generated, args.seeds, output_dir / "montage.png")

    run_metadata = {
        **metadata,
        "checkpoint": os.path.abspath(args.checkpoint),
        "reference_image": os.path.abspath(args.reference_image),
        "class_label": args.class_label,
        "seeds": args.seeds,
        "mode": args.mode,
        "num_steps": args.num_steps,
        "cfg_scale": args.cfg_scale,
        "vae": args.vae,
        "dtype": args.dtype,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(run_metadata, handle, indent=2)

    print(f"Saved {len(generated)} oracle-conditioned samples to {output_dir}")
    print(f"Montage: {output_dir / 'montage.png'}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Sample a 30K hidden or K/V checkpoint with a fixed clean-image scaffold"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--class-label", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", default="oracle_hidden_replacement")
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--heun", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--guidance-low", type=float, default=0.0)
    parser.add_argument("--guidance-high", type=float, default=1.0)
    parser.add_argument("--sde-seed", type=int, default=0)
    parser.add_argument("--vae", choices=["mse", "ema"], default="mse")
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
