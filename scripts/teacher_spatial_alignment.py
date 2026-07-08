#!/usr/bin/env python3
"""
Compute teacher-referenced spatial alignment for early checkpoints.

TRSA asks whether a SiT hidden layer preserves the teacher/scaffold token
geometry:

    corr(vec(cos(h_i, h_j)), vec(cos(e_i, e_j)))

The distance-controlled variant subtracts the mean similarity at each
Manhattan distance before computing the correlation. This makes the score less
explainable by pure local smoothness.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from q_scaffold_probe import (  # noqa: E402
    build_dataset,
    get_attr,
    load_latent_stats,
    load_model_from_checkpoint,
    make_condition_labels,
    make_noisy_model_input,
    patch_shuffle_image,
    resolve_patch_shuffle,
    sample_posterior,
    subset_dataset,
    unpack_batch,
)
from models.encoder_adapter import EncoderKVExtractor  # noqa: E402
from vision_encoder import load_encoders  # noqa: E402


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


def parse_csv_ints(values: Optional[str], default: Iterable[int]) -> List[int]:
    if values is None or not str(values).strip():
        return list(default)
    return [int(v.strip()) for v in str(values).split(",") if v.strip()]


def parse_csv_floats(values: str) -> List[float]:
    return [float(v.strip()) for v in str(values).split(",") if v.strip()]


def pair_indices(num_tokens: int, device: torch.device):
    return torch.triu_indices(num_tokens, num_tokens, offset=1, device=device)


def pair_distances(num_tokens: int, device: torch.device) -> torch.Tensor:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Token count {num_tokens} is not a square")
    yy, xx = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    coords = torch.stack([yy.flatten(), xx.flatten()], dim=1)
    dist = (coords[:, None, :] - coords[None, :, :]).abs().sum(-1)
    row, col = pair_indices(num_tokens, device)
    return dist[row, col]


def resize_token_grid(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Interpolate square-grid token features to match the SiT token grid."""
    if tokens.size(1) == target_tokens:
        return tokens

    source_tokens = tokens.size(1)
    source_side = int(math.sqrt(source_tokens))
    target_side = int(math.sqrt(target_tokens))
    if source_side * source_side != source_tokens:
        raise ValueError(f"Teacher token count {source_tokens} is not a square")
    if target_side * target_side != target_tokens:
        raise ValueError(f"Target token count {target_tokens} is not a square")

    grid = tokens.transpose(1, 2).reshape(
        tokens.size(0),
        tokens.size(2),
        source_side,
        source_side,
    )
    grid = F.interpolate(
        grid.float(),
        size=(target_side, target_side),
        mode="bilinear",
        align_corners=False,
    )
    return grid.flatten(2).transpose(1, 2).to(tokens.dtype)


def pairwise_cosine_vector(tokens: torch.Tensor) -> torch.Tensor:
    tokens = F.normalize(tokens.float(), dim=-1, eps=1e-8)
    gram = torch.bmm(tokens, tokens.transpose(1, 2))
    row, col = pair_indices(tokens.size(1), tokens.device)
    return gram[:, row, col]


def pearson_per_sample(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x0 = x - x.mean(dim=1, keepdim=True)
    y0 = y - y.mean(dim=1, keepdim=True)
    denom = x0.norm(dim=1) * y0.norm(dim=1)
    return (x0 * y0).sum(dim=1) / denom.clamp(min=eps)


def residualize_by_distance(values: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    out = values.clone()
    for d in distances.unique(sorted=True):
        mask = distances == d
        if mask.any():
            out[:, mask] = out[:, mask] - out[:, mask].mean(dim=1, keepdim=True)
    return out


@torch.no_grad()
def compute_trsa_metrics(
    dit_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
) -> Dict[str, float]:
    teacher_tokens = resize_token_grid(teacher_tokens, dit_tokens.size(1))

    dit_vec = pairwise_cosine_vector(dit_tokens)
    teacher_vec = pairwise_cosine_vector(teacher_tokens)

    trsa = pearson_per_sample(dit_vec, teacher_vec)

    distances = pair_distances(dit_tokens.size(1), dit_tokens.device)
    dit_resid = residualize_by_distance(dit_vec, distances)
    teacher_resid = residualize_by_distance(teacher_vec, distances)
    trsa_dc = pearson_per_sample(dit_resid, teacher_resid)

    pairwise_mse = (dit_vec - teacher_vec).square().mean(dim=1)

    return {
        "trsa_pearson": trsa.mean().item(),
        "trsa_dc_pearson": trsa_dc.mean().item(),
        "pairwise_mse_to_teacher": pairwise_mse.mean().item(),
        "dit_pairwise_std": dit_vec.std(dim=1, unbiased=False).mean().item(),
        "teacher_pairwise_std": teacher_vec.std(dim=1, unbiased=False).mean().item(),
    }


def select_effective_eval_mode(eval_mode: str, meta: Dict[str, object]) -> str:
    if eval_mode in ("native", "scaffold"):
        return eval_mode
    if eval_mode != "auto":
        raise ValueError(f"Unknown eval mode: {eval_mode}")

    step = int(meta["step"])
    stage1_steps = int(meta.get("stage1_steps", 30000))
    return "scaffold" if step <= stage1_steps else "native"


def cast_enc_kv(
    enc_kv,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_raw, k_raw, v_raw = enc_kv
    return (
        q_raw.to(device=device, dtype=dtype),
        k_raw.to(device=device, dtype=dtype),
        v_raw.to(device=device, dtype=dtype),
    )


@torch.no_grad()
def extract_hidden_states_by_layer(
    model,
    x_t: torch.Tensor,
    t: torch.Tensor,
    labels: torch.Tensor,
    layer_depths: Iterable[int],
    enc_kv_list=None,
    stage: int = 2,
) -> Dict[int, torch.Tensor]:
    target_layers = set(int(v) for v in layer_depths)
    max_layer = max(target_layers)

    x_tokens = model.x_embedder(x_t) + model.pos_embed
    t_embed = model.t_embedder(t)
    y_embed = model.y_embedder(labels, False)
    c = t_embed + y_embed

    hidden_by_layer: Dict[int, torch.Tensor] = {}
    for layer_idx, block in enumerate(model.blocks, start=1):
        enc_kv = None
        layer_idx0 = layer_idx - 1
        if enc_kv_list is not None and layer_idx0 in getattr(model, "sit_to_enc_idx", {}):
            enc_idx = model.sit_to_enc_idx[layer_idx0]
            if enc_idx < len(enc_kv_list):
                enc_kv = cast_enc_kv(enc_kv_list[enc_idx], x_tokens.device, x_tokens.dtype)

        old_training = block.training
        if enc_kv is not None and stage == 1:
            # SiTBlockWithEncoderKV gates encoder projection on block.training.
            # Flip only this flag so eval-time submodules stay in eval mode.
            block.training = True
        try:
            try:
                block_out = block(x_tokens, c, enc_kv=enc_kv, stage=stage)
            except TypeError:
                block_out = block(x_tokens, c)
        finally:
            block.training = old_training
        x_tokens = block_out[0] if isinstance(block_out, tuple) else block_out

        if layer_idx in target_layers:
            hidden_by_layer[layer_idx] = x_tokens.float()
        if layer_idx >= max_layer:
            break

    return hidden_by_layer


def prepare_encoder_input(
    encoder,
    raw_image: torch.Tensor,
    use_patch_shuffle: bool,
    patch_shuffle_grid: int,
    patch_shuffle_patch_size: int,
) -> torch.Tensor:
    raw_image_enc = encoder.preprocess(raw_image)
    if use_patch_shuffle:
        raw_image_enc = patch_shuffle_image(
            raw_image_enc,
            grid=patch_shuffle_grid,
            patch_size=patch_shuffle_patch_size,
        )
    return raw_image_enc


def get_teacher_tokens_from_preprocessed(
    encoder,
    raw_image_enc: torch.Tensor,
) -> torch.Tensor:
    features = encoder.forward_features(raw_image_enc)
    tokens = features.get("x_norm_patchtokens")
    if tokens is None:
        raise ValueError("Encoder did not return x_norm_patchtokens")
    return tokens.float()


@torch.no_grad()
def evaluate_teacher_spatial_alignment(args):
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

    timesteps = parse_csv_floats(args.timesteps)
    layer_depths = parse_csv_ints(args.layer_depths, default=[8, 10])

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

    teacher_type = args.teacher_enc_type or str(meta["enc_type"]).split(",")[0]
    print("Loading teacher encoder for spatial alignment")
    encoder = load_encoders(
        teacher_type,
        device,
        int(meta["resolution"]),
        accelerator=None,
    )[0]
    effective_eval_mode = select_effective_eval_mode(args.eval_mode, meta)

    use_patch_shuffle = resolve_patch_shuffle(meta, args.patch_shuffle_mode)
    use_scaffold_patch_shuffle = resolve_patch_shuffle(
        meta,
        args.scaffold_patch_shuffle_mode,
    )
    extractor = None
    if effective_eval_mode == "scaffold":
        enc_layer_indices_0 = [idx - 1 for idx in meta["enc_layer_indices"]]
        extractor = EncoderKVExtractor(encoder.model, enc_layer_indices_0)
        extractor._target_num_patches = model.x_embedder.num_patches
        extractor.eval()

    print("\nTeacher spatial alignment setup")
    print(f"  checkpoint={args.checkpoint}")
    print(f"  teacher={teacher_type}")
    print(f"  eval_mode={args.eval_mode} effective={effective_eval_mode}")
    print(f"  layers={layer_depths} timesteps={timesteps}")
    print(f"  samples={len(dataset)} conditioning={args.conditioning}")
    print(f"  teacher_patch_shuffle={use_patch_shuffle} mode={args.patch_shuffle_mode}")
    print(
        "  scaffold_patch_shuffle="
        f"{use_scaffold_patch_shuffle} mode={args.scaffold_patch_shuffle_mode}"
    )

    accum = {
        f"t={t_val:g}": {f"layer_{layer}": MeanAccumulator() for layer in layer_depths}
        for t_val in timesteps
    }

    total_seen = 0
    try:
        for batch in tqdm(dataloader, desc="teacher-align"):
            raw_image, moments, labels = unpack_batch(batch)
            raw_image = raw_image.to(device, non_blocking=True)
            moments = moments.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            bsz = labels.size(0)

            teacher_image_enc = prepare_encoder_input(
                encoder,
                raw_image,
                use_patch_shuffle=use_patch_shuffle,
                patch_shuffle_grid=int(meta["encoder_patch_shuffle_grid"]),
                patch_shuffle_patch_size=int(meta["encoder_patch_shuffle_patch_size"]),
            )
            teacher_tokens = get_teacher_tokens_from_preprocessed(
                encoder,
                teacher_image_enc,
            )

            enc_kv_list = None
            if extractor is not None:
                if use_scaffold_patch_shuffle == use_patch_shuffle:
                    scaffold_image_enc = teacher_image_enc
                else:
                    scaffold_image_enc = prepare_encoder_input(
                        encoder,
                        raw_image,
                        use_patch_shuffle=use_scaffold_patch_shuffle,
                        patch_shuffle_grid=int(meta["encoder_patch_shuffle_grid"]),
                        patch_shuffle_patch_size=int(meta["encoder_patch_shuffle_patch_size"]),
                    )
                enc_kv_list, _ = extractor(scaffold_image_enc)

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

            stage = 1 if effective_eval_mode == "scaffold" else 2
            for t_val in timesteps:
                t = torch.full((bsz,), t_val, device=device, dtype=model_dtype)
                x_t = make_noisy_model_input(
                    x0,
                    t,
                    path_type=str(meta["path_type"]),
                )
                hidden_by_layer = extract_hidden_states_by_layer(
                    model,
                    x_t,
                    t,
                    cond_labels,
                    layer_depths,
                    enc_kv_list=enc_kv_list,
                    stage=stage,
                )

                for layer in layer_depths:
                    if layer not in hidden_by_layer:
                        continue
                    metrics = compute_trsa_metrics(hidden_by_layer[layer], teacher_tokens)
                    accum[f"t={t_val:g}"][f"layer_{layer}"].update(metrics, bsz)

            total_seen += bsz
            if args.num_samples > 0 and total_seen >= args.num_samples:
                break
    finally:
        if extractor is not None:
            extractor.remove_hooks()

    results = {
        t_key: {
            layer_key: layer_accum.mean()
            for layer_key, layer_accum in layer_accums.items()
        }
        for t_key, layer_accums in accum.items()
    }

    meta = dict(meta)
    meta["teacher_enc_type"] = teacher_type
    meta["teacher_patch_shuffle"] = use_patch_shuffle
    meta["scaffold_patch_shuffle"] = use_scaffold_patch_shuffle
    meta["trsa_eval_mode"] = args.eval_mode
    meta["trsa_effective_eval_mode"] = effective_eval_mode

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).resolve().parent.parent / "teacher_spatial_alignment"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"teacher_spatial_alignment_{Path(args.checkpoint).stem}.json"
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
    parser = argparse.ArgumentParser(
        description="Compute teacher-referenced spatial alignment (TRSA)"
    )
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
    parser.add_argument(
        "--layer-depths",
        type=str,
        default="8,10",
        help=(
            "Comma-separated SiT layers. Layer 8 is not directly touched by "
            "REPA; layer 10 is the alignment layer and should be interpreted "
            "with that caveat."
        ),
    )
    parser.add_argument("--conditioning", choices=["data", "null", "zero"], default="data")
    parser.add_argument(
        "--teacher-enc-type",
        type=str,
        default=None,
        help="Teacher encoder type. Defaults to the first encoder in the checkpoint.",
    )
    parser.add_argument(
        "--eval-mode",
        choices=["auto", "native", "scaffold"],
        default="auto",
        help=(
            "auto uses scaffold K/V for checkpoints at or before stage1_steps "
            "and native K/V afterwards. native always disables scaffold. "
            "scaffold always injects encoder K/V."
        ),
    )
    parser.add_argument(
        "--patch-shuffle-mode",
        choices=["checkpoint", "on", "off"],
        default="off",
        help=(
            "Patch-shuffle the teacher reference input. Default off keeps the "
            "reference as canonical DINO image geometry."
        ),
    )
    parser.add_argument(
        "--scaffold-patch-shuffle-mode",
        choices=["checkpoint", "on", "off"],
        default="checkpoint",
        help=(
            "Patch-shuffle mode for injected scaffold K/V. Default checkpoint "
            "matches the run config while the reference can remain canonical."
        ),
    )
    evaluate_teacher_spatial_alignment(parser.parse_args())
