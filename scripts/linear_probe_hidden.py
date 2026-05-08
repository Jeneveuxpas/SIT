#!/usr/bin/env python3
"""
Linear-probe ImageNet classification from frozen SiT hidden states.

The ImageNet label is used only as the classifier target. During SiT feature
extraction, the model is conditioned on a fixed null/dummy label to avoid label
leakage from the class embedding.

Example:
    CUDA_VISIBLE_DEVICES=0 python scripts/linear_probe_hidden.py \
        --checkpoint exps/SIT-XL/checkpoints/0100000.pt \
        --data-dir /dev/shm/data \
        --layer-depth 8 \
        --timestep 0.0 \
        --epochs 20 \
        --batch-size 256
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dataset import HFImgLatentDataset, ImageFolderLatentDataset  # noqa: E402
from models.sit import SiT_models  # noqa: E402


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


def load_latent_stats(device: torch.device, vae_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    stats_path = Path(__file__).resolve().parent.parent / "pretrained_models" / (
        f"sdvae-ft-{vae_name}-f8d4-latents-stats.pt"
    )
    if not stats_path.exists():
        raise FileNotFoundError(f"Latent stats not found: {stats_path}")

    stats = torch.load(stats_path, map_location=device, weights_only=False)
    latents_scale = stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return latents_scale, latents_bias


def get_attr(obj, name: str, default=None):
    return getattr(obj, name, default) if obj is not None else default


def strip_encoderkv_name(model_name: str) -> str:
    return model_name.replace("-EncoderKV", "")


def load_sit_for_features(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype,
    override_model: Optional[str] = None,
) -> Tuple[nn.Module, object, Dict[str, object]]:
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None)

    model_name = override_model or strip_encoderkv_name(get_attr(ckpt_args, "model", "SiT-B/2"))
    if model_name not in SiT_models:
        raise ValueError(f"Unknown SiT model '{model_name}'. Available: {sorted(SiT_models)}")

    resolution = int(get_attr(ckpt_args, "resolution", 256))
    latent_size = resolution // 8
    num_classes = int(get_attr(ckpt_args, "num_classes", 1000))
    cfg_prob = float(get_attr(ckpt_args, "cfg_prob", 0.1))
    qk_norm = bool(get_attr(ckpt_args, "qk_norm", False))
    fused_attn = bool(get_attr(ckpt_args, "fused_attn", True))
    encoder_depth = int(get_attr(ckpt_args, "encoder_depth", 8))

    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=4,
        num_classes=num_classes,
        use_cfg=cfg_prob > 0,
        class_dropout_prob=cfg_prob,
        encoder_depth=encoder_depth,
        eval_mode=True,
        fused_attn=fused_attn,
        qk_norm=qk_norm,
    ).to(device=device, dtype=dtype)

    state_dict = ckpt.get("ema", ckpt.get("model", ckpt))
    filtered = {}
    ignored = []
    for key, value in state_dict.items():
        if "kv_proj" in key or "projector" in key:
            ignored.append(key)
            continue
        filtered[key] = value

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    print(
        f"Loaded {model_name}: {len(missing)} missing, "
        f"{len(unexpected)} unexpected, {len(ignored)} ignored projector/KV keys"
    )
    if missing:
        print(f"  Missing sample: {missing[:8]}")
    if unexpected:
        print(f"  Unexpected sample: {unexpected[:8]}")

    model.eval()
    model.requires_grad_(False)
    meta = {
        "model": model_name,
        "resolution": resolution,
        "num_classes": num_classes,
        "path_type": get_attr(ckpt_args, "path_type", "linear"),
        "cfg_prob": cfg_prob,
    }
    return model, ckpt_args, meta


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


def maybe_subset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_samples].tolist()
    return Subset(dataset, indices)


def unpack_batch(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    if len(batch) == 3:
        _, moments, labels = batch
    elif len(batch) == 2:
        moments, labels = batch
    else:
        raise ValueError(f"Unsupported batch with {len(batch)} elements")
    return moments, labels


def make_condition_labels(
    model: nn.Module,
    labels: torch.Tensor,
    num_classes: int,
    conditioning: str,
) -> torch.Tensor:
    if conditioning == "zero":
        return torch.zeros_like(labels)
    if conditioning != "null":
        raise ValueError(f"Unknown conditioning: {conditioning}")

    emb = model.y_embedder.embedding_table
    if emb.num_embeddings > num_classes:
        return torch.full_like(labels, num_classes)

    return torch.zeros_like(labels)


def pool_tokens(tokens: torch.Tensor, pool: str) -> torch.Tensor:
    if pool == "mean":
        return tokens.mean(dim=1)
    if pool == "max":
        return tokens.max(dim=1).values
    if pool == "meanmax":
        return torch.cat([tokens.mean(dim=1), tokens.max(dim=1).values], dim=-1)
    raise ValueError(f"Unknown pool mode: {pool}")


@torch.no_grad()
def extract_features(
    model: nn.Module,
    moments: torch.Tensor,
    labels: torch.Tensor,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    layer_depth: int,
    timestep: float,
    path_type: str,
    latent_mode: str,
    pool: str,
    conditioning: str,
    num_classes: int,
    model_dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    moments = moments.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True).long()

    x0 = sample_posterior(moments, latents_scale, latents_bias, mode=latent_mode).to(model_dtype)
    t = torch.full((x0.size(0),), timestep, device=device, dtype=model_dtype)
    x_t = make_noisy_model_input(x0, t, path_type=path_type)
    cond_y = make_condition_labels(model, labels, num_classes, conditioning)

    hidden_list = model.forward_features(
        x_t,
        t,
        cond_y,
        encoder_depths=[layer_depth],
        proj=False,
    )
    if not hidden_list:
        raise ValueError(
            f"Layer depth {layer_depth} was not returned. "
            f"Model has {len(model.blocks)} transformer blocks."
        )
    hidden = hidden_list[0]
    features = pool_tokens(hidden, pool).to(torch.float32)
    return features, labels


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, topk: Iterable[int] = (1, 5)):
    maxk = max(topk)
    _, pred = logits.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum().item() for k in topk]


def run_epoch(
    model: nn.Module,
    classifier: nn.Module,
    dataloader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    args,
    meta: Dict[str, object],
    device: torch.device,
    model_dtype: torch.dtype,
    desc: str,
) -> Dict[str, float]:
    train = optimizer is not None
    classifier.train(train)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total = 0

    pbar = tqdm(dataloader, desc=desc)
    for batch in pbar:
        moments, labels = unpack_batch(batch)
        features, labels = extract_features(
            model=model,
            moments=moments,
            labels=labels,
            latents_scale=latents_scale,
            latents_bias=latents_bias,
            layer_depth=args.layer_depth,
            timestep=args.timestep,
            path_type=meta["path_type"],
            latent_mode=args.latent_mode,
            pool=args.pool,
            conditioning=args.conditioning,
            num_classes=meta["num_classes"],
            model_dtype=model_dtype,
            device=device,
        )

        if train:
            logits = classifier(features)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = classifier(features)
                loss = F.cross_entropy(logits, labels)

        bs = labels.size(0)
        top1, top5 = topk_accuracy(logits.detach(), labels, topk=(1, 5))
        total_loss += loss.item() * bs
        total_top1 += top1
        total_top5 += top5
        total += bs
        pbar.set_postfix(
            loss=f"{total_loss / total:.4f}",
            top1=f"{100 * total_top1 / total:.2f}",
            top5=f"{100 * total_top5 / total:.2f}",
        )

    return {
        "loss": total_loss / max(1, total),
        "top1": 100 * total_top1 / max(1, total),
        "top5": 100 * total_top5 / max(1, total),
        "num_samples": total,
    }


def infer_feature_dim(
    model: nn.Module,
    dataloader: DataLoader,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    args,
    meta: Dict[str, object],
    device: torch.device,
    model_dtype: torch.dtype,
) -> int:
    batch = next(iter(dataloader))
    moments, labels = unpack_batch(batch)
    features, _ = extract_features(
        model=model,
        moments=moments,
        labels=labels,
        latents_scale=latents_scale,
        latents_bias=latents_bias,
        layer_depth=args.layer_depth,
        timestep=args.timestep,
        path_type=meta["path_type"],
        latent_mode=args.latent_mode,
        pool=args.pool,
        conditioning=args.conditioning,
        num_classes=meta["num_classes"],
        model_dtype=model_dtype,
        device=device,
    )
    return features.size(-1)


def build_optimizer(classifier: nn.Module, args):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            classifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            classifier.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unknown optimizer: {args.optimizer}")


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

    model, _, meta = load_sit_for_features(
        args.checkpoint,
        device=device,
        dtype=model_dtype,
        override_model=args.model,
    )
    if args.path_type is not None:
        meta["path_type"] = args.path_type

    latents_scale, latents_bias = load_latent_stats(device, args.vae)

    train_dataset = build_dataset(args.data_dir, args.vae, "train", meta["resolution"])
    val_dataset = build_dataset(args.data_dir, args.vae, "val", meta["resolution"])
    train_dataset = maybe_subset(train_dataset, args.max_train_samples, args.seed)
    val_dataset = maybe_subset(val_dataset, args.max_val_samples, args.seed + 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    feature_dim = infer_feature_dim(
        model, train_loader, latents_scale, latents_bias, args, meta, device, model_dtype
    )
    classifier = nn.Linear(feature_dim, meta["num_classes"]).to(device)
    optimizer = build_optimizer(classifier, args)

    print("\nLinear probe setup")
    print(f"  model={meta['model']} checkpoint={args.checkpoint}")
    print(f"  layer_depth={args.layer_depth} timestep={args.timestep} pool={args.pool}")
    print(f"  conditioning={args.conditioning} latent_mode={args.latent_mode}")
    print(f"  feature_dim={feature_dim} train={len(train_dataset)} val={len(val_dataset)}")
    print(f"  optimizer={args.optimizer} lr={args.learning_rate} wd={args.weight_decay}")

    best = {"top1": -1.0, "epoch": -1}
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            classifier,
            train_loader,
            optimizer,
            latents_scale,
            latents_bias,
            args,
            meta,
            device,
            model_dtype,
            desc=f"train {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model,
            classifier,
            val_loader,
            None,
            latents_scale,
            latents_bias,
            args,
            meta,
            device,
            model_dtype,
            desc=f"val {epoch}/{args.epochs}",
        )
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch}: "
            f"train top1={train_metrics['top1']:.2f} top5={train_metrics['top5']:.2f} "
            f"val top1={val_metrics['top1']:.2f} top5={val_metrics['top5']:.2f}"
        )

        if val_metrics["top1"] > best["top1"]:
            best = {"top1": val_metrics["top1"], "epoch": epoch, "metrics": val_metrics}
            if args.save_classifier:
                output_dir = Path(args.output_dir) if args.output_dir else (
                    Path(args.checkpoint).resolve().parent.parent / "linear_probe"
                )
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "classifier": classifier.state_dict(),
                        "feature_dim": feature_dim,
                        "args": vars(args),
                        "meta": meta,
                        "best": best,
                    },
                    output_dir / "best_classifier.pt",
                )

    result = {
        "checkpoint": args.checkpoint,
        "meta": meta,
        "args": vars(args),
        "feature_dim": feature_dim,
        "best": best,
        "history": history,
    }
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.checkpoint).resolve().parent.parent / "linear_probe"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / (
        f"linear_probe_{Path(args.checkpoint).stem}"
        f"_layer{args.layer_depth}_t{args.timestep:g}_{args.pool}.json"
    )
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBest val top1={best['top1']:.2f} at epoch {best['epoch']}")
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear probe on frozen SiT hidden states")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="/dev/shm/data")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inference-dtype", choices=["fp32", "bf16", "fp16"], default="fp32")

    parser.add_argument("--vae", choices=["mse"], default="mse")
    parser.add_argument("--latent-mode", choices=["mean", "sample"], default="mean")
    parser.add_argument("--timestep", type=float, default=0.0)
    parser.add_argument("--path-type", choices=["linear", "cosine"], default=None)
    parser.add_argument("--layer-depth", type=int, default=8)
    parser.add_argument("--pool", choices=["mean", "max", "meanmax"], default="mean")
    parser.add_argument("--conditioning", choices=["null", "zero"], default="null")

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--save-classifier", action="store_true")
    main(parser.parse_args())
