#!/usr/bin/env python3
"""
Probe how native SiT queries retrieve the encoder scaffold.

For each checkpoint this script:
  1. Loads the EncoderKV SiT checkpoint.
  2. Extracts encoder Q/K/V on a fixed image subset.
  3. Projects encoder K/V through the checkpoint's kv_proj.
  4. Measures P_scaf = softmax(Q_sit K_scaf^T) at the scaffold injection layer.

The main curve to plot is rcs_local_far: local scaffold-attention mass minus
far scaffold-attention mass. The companion metrics explain whether the score is
coming from sharper retrieval, local retrieval, or native/scaffold agreement.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import HFImgLatentDataset, ImageFolderLatentDataset  # noqa: E402
from models.encoder_adapter import EncoderKVExtractor  # noqa: E402
from models.sit import modulate  # noqa: E402
from models.sit_encoder import SiT_EncoderKV_models  # noqa: E402
from vision_encoder import load_encoders  # noqa: E402


def get_attr(obj, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def parse_indices_1based(values) -> List[int]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [int(v) - 1 for v in values]
    return [int(v.strip()) - 1 for v in str(values).split(",") if v.strip()]


def parse_weights(values, expected_len: int) -> Optional[List[float]]:
    if values is None:
        return None
    weights = [float(v.strip()) for v in str(values).split(",") if v.strip()]
    if len(weights) != expected_len:
        raise ValueError(
            f"Expected {expected_len} sit layer weights, got {len(weights)}"
        )
    total = sum(weights)
    if total <= 0:
        raise ValueError("sit layer weights must sum to a positive value")
    return [w / total for w in weights]


def sample_posterior(
    moments: torch.Tensor,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    mode: str = "mean",
) -> torch.Tensor:
    if moments.ndim == 5 and moments.size(1) == 1:
        moments = moments.squeeze(1)
    mean, std = torch.chunk(moments, 2, dim=1)
    if mode == "mean":
        z = mean
    elif mode == "sample":
        z = mean + std * torch.randn_like(mean)
    else:
        raise ValueError(f"Unknown latent mode: {mode}")
    return (z - latents_bias) * latents_scale


def make_noisy_model_input(
    x0: torch.Tensor,
    t: torch.Tensor,
    path_type: str = "linear",
) -> torch.Tensor:
    if t.ndim == 1:
        t = t.view(-1, 1, 1, 1)
    if float(t.max().item()) == 0.0:
        return x0

    noise = torch.randn_like(x0)
    if path_type == "linear":
        alpha_t = 1 - t
        sigma_t = t
    elif path_type == "cosine":
        alpha_t = torch.cos(t * math.pi / 2)
        sigma_t = torch.sin(t * math.pi / 2)
    else:
        raise ValueError(f"Unsupported path_type: {path_type}")
    return alpha_t * x0 + sigma_t * noise


def patch_shuffle_image(
    x: torch.Tensor,
    grid: int = 0,
    patch_size: int = 14,
) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected BCHW image tensor, got {tuple(x.shape)}")

    bsz, channels, height, width = x.shape
    if grid <= 0:
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"Cannot infer patch grid from {height}x{width} and patch={patch_size}"
            )
        grid_h, grid_w = height // patch_size, width // patch_size
    else:
        grid_h = grid_w = grid

    if height % grid_h != 0 or width % grid_w != 0:
        raise ValueError(f"Patch grid {grid_h}x{grid_w} must divide {height}x{width}")

    ph, pw = height // grid_h, width // grid_w
    patches = x.unfold(2, ph, ph).unfold(3, pw, pw)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.reshape(bsz, grid_h * grid_w, channels, ph, pw)

    perm = torch.stack(
        [torch.randperm(grid_h * grid_w, device=x.device) for _ in range(bsz)],
        dim=0,
    )
    batch_idx = torch.arange(bsz, device=x.device)[:, None]
    patches = patches[batch_idx, perm]

    x_shuf = patches.reshape(bsz, grid_h, grid_w, channels, ph, pw)
    x_shuf = x_shuf.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x_shuf.reshape(bsz, channels, height, width)


def load_latent_stats(
    device: torch.device,
    vae_name: str,
    latent_stats: Optional[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if latent_stats is None:
        latent_stats = (
            Path(__file__).resolve().parent.parent
            / "pretrained_models"
            / f"sdvae-ft-{vae_name}-f8d4-latents-stats.pt"
        )
    stats = torch.load(latent_stats, map_location=device, weights_only=False)
    latents_scale = stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return latents_scale, latents_bias


def build_dataset(data_dir: str, vae_name: str, split: str, resolution: int):
    try:
        return HFImgLatentDataset(f"sdvae-ft-{vae_name}-f8d4", data_dir, split=split)
    except Exception as exc:
        print(f"[warn] HFImgLatentDataset({split}) failed: {exc}")
        print(f"[warn] Falling back to ImageFolderLatentDataset({split})")
        return ImageFolderLatentDataset(
            f"sdvae-ft-{vae_name}-f8d4",
            data_dir,
            resolution=resolution,
            split=split,
        )


def subset_dataset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if len(batch) != 3:
        raise ValueError("q_scaffold_probe requires image+latent batches")
    raw_image, moments, labels = batch
    return raw_image, moments, labels


def make_condition_labels(
    model,
    labels: torch.Tensor,
    num_classes: int,
    conditioning: str,
) -> torch.Tensor:
    if conditioning == "data":
        return labels
    if conditioning == "zero":
        return torch.zeros_like(labels)
    if conditioning != "null":
        raise ValueError(f"Unknown conditioning: {conditioning}")

    emb = model.y_embedder.embedding_table
    if emb.num_embeddings > num_classes:
        return torch.full_like(labels, num_classes)
    return torch.zeros_like(labels)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    use_ema: bool,
    override_model: Optional[str],
):
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)
    if ckpt_args is None:
        raise ValueError("Checkpoint does not contain args")

    model_name = override_model or get_attr(ckpt_args, "model", "SiT-XL/2-EncoderKV")
    if model_name not in SiT_EncoderKV_models:
        raise ValueError(
            f"q_scaffold_probe requires an EncoderKV model, got {model_name}"
        )

    resolution = int(get_attr(ckpt_args, "resolution", 256))
    latent_size = resolution // 8
    enc_layer_indices = parse_indices_1based(get_attr(ckpt_args, "enc_layer_indices", "12"))
    sit_layer_indices = parse_indices_1based(get_attr(ckpt_args, "sit_layer_indices", "4"))
    sit_layer_loss_weights = parse_weights(
        get_attr(ckpt_args, "sit_layer_loss_weights", None),
        len(sit_layer_indices),
    )

    model = SiT_EncoderKV_models[model_name](
        path_type=get_attr(ckpt_args, "path_type", "linear"),
        input_size=latent_size,
        in_channels=4,
        num_classes=int(get_attr(ckpt_args, "num_classes", 1000)),
        use_cfg=float(get_attr(ckpt_args, "cfg_prob", 0.1)) > 0,
        z_dims=[768],
        encoder_depth=int(get_attr(ckpt_args, "encoder_depth", 10)),
        eval_mode=True,
        projection_layer_type=get_attr(ckpt_args, "projection_layer_type", "mlp"),
        proj_kwargs_kernel_size=int(get_attr(ckpt_args, "proj_kwargs_kernel_size", 3)),
        enc_layer_indices=enc_layer_indices,
        sit_layer_indices=sit_layer_indices,
        sit_layer_loss_weights=sit_layer_loss_weights,
        enc_dim=int(get_attr(ckpt_args, "enc_dim", 768) or 768),
        enc_heads=int(get_attr(ckpt_args, "enc_heads", 12) or 12),
        kv_proj_type=get_attr(ckpt_args, "kv_proj_type", "linear"),
        kv_proj_hidden_dim=get_attr(ckpt_args, "kv_proj_hidden_dim", None),
        kv_proj_kernel_size=int(get_attr(ckpt_args, "kv_proj_kernel_size", 1)),
        kv_norm_type=get_attr(ckpt_args, "kv_norm_type", "none"),
        kv_zscore_alpha=float(get_attr(ckpt_args, "kv_zscore_alpha", 1.0)),
        kv_replace_mode=get_attr(ckpt_args, "kv_replace_mode", "kv"),
        kv_use_adaln=bool(get_attr(ckpt_args, "kv_use_adaln", False)),
        train_kv_proj_in_stage2=bool(get_attr(ckpt_args, "train_kv_proj_stage2", False)),
        distill_temperature=float(get_attr(ckpt_args, "distill_temperature", 1.0)),
        kv_distill_snr_gamma=float(get_attr(ckpt_args, "kv_distill_snr_gamma", 1.0)),
        kv_distill_min_weight=float(get_attr(ckpt_args, "kv_distill_min_weight", 0.0)),
        attn_loss_weight=float(get_attr(ckpt_args, "attn_loss_weight", 1.0)),
        kv_loss_weight=float(get_attr(ckpt_args, "kv_loss_weight", 1.0)),
        fused_attn=bool(get_attr(ckpt_args, "fused_attn", True)),
        qk_norm=bool(get_attr(ckpt_args, "qk_norm", False)),
    )

    state_key = "ema" if use_ema and "ema" in ckpt else "model"
    state_dict = ckpt.get(state_key, ckpt)
    filtered = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("projectors.")
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded {model_name} ({state_key}): {len(missing)} missing, "
        f"{len(unexpected)} unexpected"
    )
    if missing:
        print(f"  Missing sample: {missing[:8]}")
    if unexpected:
        print(f"  Unexpected sample: {unexpected[:8]}")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)

    meta = {
        "model": model_name,
        "resolution": resolution,
        "num_classes": int(get_attr(ckpt_args, "num_classes", 1000)),
        "path_type": get_attr(ckpt_args, "path_type", "linear"),
        "enc_type": get_attr(ckpt_args, "enc_type", "dinov2-b"),
        "enc_layer_indices": [idx + 1 for idx in enc_layer_indices],
        "sit_layer_indices": [idx + 1 for idx in sit_layer_indices],
        "kv_replace_mode": get_attr(ckpt_args, "kv_replace_mode", "kv"),
        "encoder_patch_shuffle": bool(get_attr(ckpt_args, "encoder_patch_shuffle", False)),
        "encoder_patch_shuffle_grid": int(get_attr(ckpt_args, "encoder_patch_shuffle_grid", 0)),
        "encoder_patch_shuffle_patch_size": int(
            get_attr(ckpt_args, "encoder_patch_shuffle_patch_size", 14)
        ),
        "repa_loss": bool(get_attr(ckpt_args, "repa_loss", False)),
        "distill_coeff": float(get_attr(ckpt_args, "distill_coeff", 0.0)),
        "stage1_steps": int(get_attr(ckpt_args, "stage1_steps", 30000)),
        "step": int(ckpt.get("steps", Path(checkpoint_path).stem)),
    }
    return model, ckpt_args, meta


def grid_distances(num_tokens: int, device: torch.device) -> torch.Tensor:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square")
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    return (coords[:, None, :] - coords[None, :, :]).abs().sum(-1).float()


@torch.no_grad()
def compute_attention_metrics(
    q_sit: torch.Tensor,
    k_sit: torch.Tensor,
    v_sit: torch.Tensor,
    k_scaf: torch.Tensor,
    v_scaf: Optional[torch.Tensor],
    scale: float,
) -> Dict[str, float]:
    q = q_sit.float()
    k_native = k_sit.float()
    v_native = v_sit.float()
    k_teacher = k_scaf.float()
    num_tokens = q.size(-2)
    dist = grid_distances(num_tokens, q.device)

    logits_scaf = (q @ k_teacher.transpose(-2, -1)) * scale
    p_scaf = logits_scaf.softmax(dim=-1)

    eps = 1e-8
    entropy = -(p_scaf * (p_scaf + eps).log()).sum(dim=-1) / math.log(num_tokens)
    diag_mass = p_scaf.diagonal(dim1=-2, dim2=-1).mean()
    top1 = p_scaf.argmax(dim=-1)
    arange = torch.arange(num_tokens, device=q.device).view(1, 1, -1)
    top1_acc = (top1 == arange).float().mean()

    local1 = dist <= 1
    local2 = dist <= 2
    far6 = dist >= 6
    local_mass_r1 = p_scaf.masked_fill(~local1, 0.0).sum(dim=-1).mean()
    local_mass_r2 = p_scaf.masked_fill(~local2, 0.0).sum(dim=-1).mean()
    far_mass_d6 = p_scaf.masked_fill(~far6, 0.0).sum(dim=-1).mean()
    expected_dist = (p_scaf * dist).sum(dim=-1).mean()
    max_dist = float(dist.max().item())

    logits_native = (q @ k_native.transpose(-2, -1)) * scale
    p_native = logits_native.softmax(dim=-1)
    native_entropy = (
        -(p_native * (p_native + eps).log()).sum(dim=-1) / math.log(num_tokens)
    ).mean()
    map_cos = F.cosine_similarity(
        p_scaf.reshape(-1, num_tokens),
        p_native.reshape(-1, num_tokens),
        dim=-1,
    ).mean()

    out = {
        "rcs_local_far": (local_mass_r2 - far_mass_d6).item(),
        "local_mass_r1": local_mass_r1.item(),
        "local_mass_r2": local_mass_r2.item(),
        "far_mass_d6": far_mass_d6.item(),
        "diag_mass": diag_mass.item(),
        "top1_same_position": top1_acc.item(),
        "expected_distance": expected_dist.item(),
        "expected_distance_norm": (expected_dist / max(1.0, max_dist)).item(),
        "entropy_norm": entropy.mean().item(),
        "native_entropy_norm": native_entropy.item(),
        "attn_map_cosine_to_native": map_cos.item(),
        "q_norm": q.norm(dim=-1).mean().item(),
        "k_scaf_norm": k_teacher.norm(dim=-1).mean().item(),
        "k_native_norm": k_native.norm(dim=-1).mean().item(),
    }

    if v_scaf is not None:
        attn_scaf = p_scaf @ v_scaf.float()
        attn_native = p_native @ v_native
        out["attn_output_mse_to_scaffold"] = F.mse_loss(
            attn_native,
            attn_scaf,
        ).item()
        out["attn_output_cosine_to_scaffold"] = F.cosine_similarity(
            attn_native.reshape(-1, attn_native.size(-1)),
            attn_scaf.reshape(-1, attn_scaf.size(-1)),
            dim=-1,
        ).mean().item()
    return out


class MeanAccumulator:
    def __init__(self):
        self.sums: Dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Dict[str, float], weight: int):
        for key, value in metrics.items():
            self.sums[key] = self.sums.get(key, 0.0) + float(value) * weight
        self.count += weight

    def mean(self) -> Dict[str, float]:
        denom = max(1, self.count)
        out = {key: value / denom for key, value in sorted(self.sums.items())}
        out["num_samples"] = self.count
        return out


@torch.no_grad()
def get_native_qkv_by_layer(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    layer_depths: Iterable[int],
):
    target_layers = set(layer_depths)
    max_layer = max(target_layers)

    x_tokens = model.x_embedder(x_t) + model.pos_embed
    t_embed = model.t_embedder(t)
    y_embed = model.y_embedder(labels, False)
    c = t_embed + y_embed

    qkv_by_layer = {}
    for layer_idx, block in enumerate(model.blocks, start=1):
        if layer_idx in target_layers:
            shift_msa, scale_msa, _, _, _, _ = block.adaLN_modulation(c).chunk(6, dim=-1)
            attn_in = modulate(block.norm1(x_tokens), shift_msa, scale_msa)
            bsz, num_tokens, channels = attn_in.shape
            qkv = block.attn.qkv(attn_in)
            qkv = qkv.reshape(
                bsz,
                num_tokens,
                3,
                block.attn.num_heads,
                block.attn.head_dim,
            )
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q_sit, k_sit, v_sit = qkv.unbind(0)
            q_sit = block.attn.q_norm(q_sit)
            k_sit = block.attn.k_norm(k_sit)
            qkv_by_layer[layer_idx] = (q_sit, k_sit, v_sit, c)

        if layer_idx >= max_layer:
            break
        x_tokens, _ = block(x_tokens, c, enc_kv=None, stage=2)

    return qkv_by_layer


@torch.no_grad()
def project_scaffold_for_layer(
    model,
    layer_depth: int,
    enc_kv_list,
    c: torch.Tensor,
    dtype: torch.dtype,
):
    layer_idx0 = layer_depth - 1
    enc_idx = model.sit_to_enc_idx.get(layer_idx0)
    if enc_idx is None or enc_idx >= len(enc_kv_list):
        return None, None, None

    block = model.blocks[layer_idx0]
    if not hasattr(block, "kv_proj"):
        return None, None, None

    q_raw, k_raw, v_raw = enc_kv_list[enc_idx]
    q_raw = q_raw.to(device=c.device, dtype=dtype)
    k_raw = k_raw.to(device=c.device, dtype=dtype)
    v_raw = v_raw.to(device=c.device, dtype=dtype)
    return block.kv_proj(q_raw, k_raw, v_raw, stage=2, c=c)


def resolve_patch_shuffle(meta: Dict[str, object], mode: str) -> bool:
    if mode == "checkpoint":
        return bool(meta["encoder_patch_shuffle"])
    if mode == "on":
        return True
    if mode == "off":
        return False
    raise ValueError(f"Unknown patch shuffle mode: {mode}")


def main(args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)

    dtype_map = {
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }
    model_dtype = dtype_map[args.inference_dtype]

    model, _, meta = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
        dtype=model_dtype,
        use_ema=args.use_ema,
        override_model=args.model,
    )

    layer_depths = (
        [int(v.strip()) for v in args.layer_depths.split(",") if v.strip()]
        if args.layer_depths
        else list(meta["sit_layer_indices"])
    )
    timesteps = [float(v.strip()) for v in args.timesteps.split(",") if v.strip()]

    latents_scale, latents_bias = load_latent_stats(device, args.vae, args.latent_stats)
    dataset = build_dataset(args.data_dir, args.vae, args.split, meta["resolution"])
    dataset = subset_dataset(dataset, args.num_samples, args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    print("Loading encoder for scaffold extraction")
    encoders = load_encoders(
        str(meta["enc_type"]),
        device,
        int(meta["resolution"]),
        accelerator=None,
    )
    encoder = encoders[0]
    enc_layer_indices_0 = [idx - 1 for idx in meta["enc_layer_indices"]]
    extractor = EncoderKVExtractor(encoder.model, enc_layer_indices_0)
    extractor._target_num_patches = model.x_embedder.num_patches
    extractor.eval()

    use_patch_shuffle = resolve_patch_shuffle(meta, args.patch_shuffle_mode)
    print("\nQ scaffold probe setup")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  layers={layer_depths} timesteps={timesteps}")
    print(f"  samples={len(dataset)} conditioning={args.conditioning}")
    print(f"  patch_shuffle={use_patch_shuffle} mode={args.patch_shuffle_mode}")

    accum = {
        f"t={t_val:g}": {f"layer_{layer}": MeanAccumulator() for layer in layer_depths}
        for t_val in timesteps
    }
    total_seen = 0

    try:
        for batch in tqdm(dataloader, desc="probe"):
            raw_image, moments, labels = unpack_batch(batch)
            raw_image = raw_image.to(device, non_blocking=True)
            moments = moments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            bsz = labels.size(0)

            raw_image_enc = encoder.preprocess(raw_image)
            if use_patch_shuffle:
                raw_image_enc = patch_shuffle_image(
                    raw_image_enc,
                    grid=int(meta["encoder_patch_shuffle_grid"]),
                    patch_size=int(meta["encoder_patch_shuffle_patch_size"]),
                )
            enc_kv_list, _ = extractor(raw_image_enc)

            x0 = sample_posterior(
                moments,
                latents_scale,
                latents_bias,
                mode=args.latent_mode,
            ).to(model_dtype)
            cond_labels = make_condition_labels(
                model,
                labels,
                int(meta["num_classes"]),
                args.conditioning,
            )

            for t_val in timesteps:
                t = torch.full((bsz,), t_val, device=device, dtype=model_dtype)
                x_t = make_noisy_model_input(
                    x0,
                    t,
                    path_type=str(meta["path_type"]),
                )
                qkv_by_layer = get_native_qkv_by_layer(
                    model,
                    x_t,
                    t,
                    cond_labels,
                    layer_depths,
                )

                for layer in layer_depths:
                    if layer not in qkv_by_layer:
                        continue
                    q_sit, k_sit, v_sit, c = qkv_by_layer[layer]
                    _, k_scaf, v_scaf = project_scaffold_for_layer(
                        model,
                        layer,
                        enc_kv_list,
                        c,
                        model_dtype,
                    )
                    if k_scaf is None:
                        continue
                    metrics = compute_attention_metrics(
                        q_sit,
                        k_sit,
                        v_sit,
                        k_scaf,
                        v_scaf,
                        scale=model.blocks[layer - 1].attn.scale,
                    )
                    accum[f"t={t_val:g}"][f"layer_{layer}"].update(metrics, bsz)

            total_seen += bsz
            if args.num_samples > 0 and total_seen >= args.num_samples:
                break
    finally:
        extractor.remove_hooks()

    results = {
        t_key: {
            layer_key: layer_accum.mean()
            for layer_key, layer_accum in layer_accums.items()
        }
        for t_key, layer_accums in accum.items()
    }

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).resolve().parent.parent / "q_scaffold_probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"q_scaffold_probe_{Path(args.checkpoint).stem}.json"
    payload = {
        "checkpoint": args.checkpoint,
        "meta": meta,
        "args": vars(args),
        "results": results,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved results to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Probe Q_sit retrieval over scaffold K")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/dev/shm/data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--vae", choices=["mse"], default="mse")
    parser.add_argument("--latent-stats", type=str, default=None)
    parser.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=12)

    parser.add_argument("--timesteps", type=str, default="0.0,0.1,0.5")
    parser.add_argument("--layer-depths", type=str, default=None)
    parser.add_argument("--conditioning", choices=["data", "null", "zero"], default="data")
    parser.add_argument(
        "--patch-shuffle-mode",
        choices=["checkpoint", "on", "off"],
        default="checkpoint",
        help="Whether to patch-shuffle encoder input during scaffold extraction.",
    )
    main(parser.parse_args())
