"""Parallel ImageNet preprocessing for the EDM2/REPA folder format.

This complements ``dataset_tools.py`` for large folder-based datasets:

* ``convert`` uses multiple CPU processes for crop/resize/PNG encoding.
* ``encode`` uses one persistent process and VAE instance per GPU.

Each worker writes a disjoint deterministic output filename.  The parent process
writes ``dataset.json`` only after every item succeeds, so images and latent
moments retain exactly the same ordering and labels.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import io
import json
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterator, Optional

import click
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm

from dataset_tools import is_image_ext, make_transform, parse_tuple
from encoders import StabilityVAEEncoder


def _discover_folder(source: str, max_images: Optional[int]) -> list[tuple[str, Optional[int]]]:
    """Return image paths and labels in the same sorted order as dataset_tools.py."""
    source = os.path.abspath(source)
    input_images: list[str] = []

    def recurse(root: str) -> None:
        with os.scandir(root) as entries:
            for entry in entries:
                path = os.path.join(root, entry.name)
                if entry.is_file() and is_image_ext(path):
                    input_images.append(path)
                elif entry.is_dir():
                    recurse(path)

    recurse(source)
    input_images.sort()
    if max_images is not None:
        input_images = input_images[:max_images]

    relative = {
        path: os.path.relpath(path, source).replace("\\", "/")
        for path in input_images
    }
    labels: dict[str, int] = {}
    metadata_path = os.path.join(source, "dataset.json")
    if os.path.isfile(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata_labels = json.load(file).get("labels")
        if metadata_labels is not None:
            labels = {name: int(label) for name, label in metadata_labels}

    if not labels:
        top_levels = {
            rel: rel.split("/", 1)[0] if "/" in rel else ""
            for rel in relative.values()
        }
        class_names = sorted(set(top_levels.values()))
        if len(class_names) > 1:
            class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            labels = {rel: class_to_idx[top] for rel, top in top_levels.items()}

    return [(path, labels.get(relative[path])) for path in input_images]


def _output_name(idx: int, latent: bool) -> str:
    idx_str = f"{idx:08d}"
    if latent:
        return f"{idx_str[:5]}/img-mean-std-{idx_str}.npy"
    return f"{idx_str[:5]}/img{idx_str}.png"


def _atomic_write(path: str, data: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp-{os.getpid()}"
    with open(temporary, "wb") as file:
        file.write(data)
    os.replace(temporary, path)


def _prepare_dest(dest: str, resume: bool) -> str:
    if Path(dest).suffix.lower() == ".zip":
        raise click.ClickException("Parallel output must be a directory, not a ZIP archive")
    dest = os.path.abspath(dest)
    if os.path.isdir(dest) and os.listdir(dest) and not resume:
        raise click.ClickException("--dest must be empty; pass --resume to continue an interrupted run")
    os.makedirs(dest, exist_ok=True)
    return dest


_CONVERT_DEST: str
_CONVERT_TRANSFORM = None
_CONVERT_RESUME: bool


def _init_convert_worker(dest: str, transform: Optional[str], resolution: Optional[tuple[int, int]], resume: bool) -> None:
    global _CONVERT_DEST, _CONVERT_TRANSFORM, _CONVERT_RESUME
    PIL.Image.init()
    _CONVERT_DEST = dest
    _CONVERT_TRANSFORM = make_transform(
        transform,
        *(resolution if resolution is not None else (None, None)),
    )
    _CONVERT_RESUME = resume


def _convert_one(task: tuple[int, str, Optional[int]]) -> tuple[str, Optional[int], tuple[int, int]]:
    idx, source_path, label = task
    archive_name = _output_name(idx, latent=False)
    output_path = os.path.join(_CONVERT_DEST, archive_name)
    if _CONVERT_RESUME and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        with PIL.Image.open(output_path) as existing:
            return archive_name, label, existing.size

    with PIL.Image.open(source_path) as source_image:
        image = np.asarray(source_image.convert("RGB"))
    image = _CONVERT_TRANSFORM(image)
    if image is None:
        raise RuntimeError(f"Transform discarded image: {source_path}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise RuntimeError(f"Unexpected transformed shape {image.shape}: {source_path}")
    height, width = image.shape[:2]
    if width != height or width != 2 ** int(np.floor(np.log2(width))):
        raise RuntimeError(f"Output must be square and power-of-two, got {width}x{height}: {source_path}")

    buffer = io.BytesIO()
    PIL.Image.fromarray(image).save(buffer, format="png", compress_level=0, optimize=False)
    _atomic_write(output_path, buffer.getvalue())
    return archive_name, label, (width, height)


_ENCODE_DEST: str
_ENCODE_RESUME: bool
_ENCODER: StabilityVAEEncoder
_ENCODE_DEVICE: torch.device


def _init_encode_worker(model_url: str, dest: str, resume: bool, device_queue) -> None:
    global _ENCODE_DEST, _ENCODE_RESUME, _ENCODER, _ENCODE_DEVICE
    PIL.Image.init()
    gpu_id = str(device_queue.get())
    torch.cuda.set_device(int(gpu_id))
    _ENCODE_DEVICE = torch.device(f"cuda:{gpu_id}")
    _ENCODE_DEST = dest
    _ENCODE_RESUME = resume
    _ENCODER = StabilityVAEEncoder(vae_name=model_url, batch_size=1024)
    _ENCODER.init(_ENCODE_DEVICE)


def _encode_batch(tasks: list[tuple[int, str, Optional[int]]]) -> list[tuple[str, Optional[int]]]:
    pending: list[tuple[int, str, Optional[int]]] = []
    results: dict[int, tuple[str, Optional[int]]] = {}
    for idx, source_path, label in tasks:
        archive_name = _output_name(idx, latent=True)
        output_path = os.path.join(_ENCODE_DEST, archive_name)
        if _ENCODE_RESUME and os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
            results[idx] = (archive_name, label)
        else:
            pending.append((idx, source_path, label))

    if pending:
        images = []
        for _, source_path, _ in pending:
            with PIL.Image.open(source_path) as source_image:
                images.append(np.asarray(source_image.convert("RGB"), dtype=np.uint8))
        image_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).to(_ENCODE_DEVICE)
        with torch.inference_mode():
            moments = _ENCODER.encode_pixels(image_tensor).cpu().numpy()

        for (idx, _, label), mean_std in zip(pending, moments):
            archive_name = _output_name(idx, latent=True)
            buffer = io.BytesIO()
            np.save(buffer, mean_std)
            _atomic_write(os.path.join(_ENCODE_DEST, archive_name), buffer.getvalue())
            results[idx] = (archive_name, label)

    return [results[idx] for idx, _, _ in tasks]


def _batches(entries: list[tuple[str, Optional[int]]], batch_size: int) -> Iterator[list[tuple[int, str, Optional[int]]]]:
    for start in range(0, len(entries), batch_size):
        yield [
            (idx, entries[idx][0], entries[idx][1])
            for idx in range(start, min(start + batch_size, len(entries)))
        ]


def _write_metadata(dest: str, rows: list[tuple[str, Optional[int]]]) -> None:
    labels = [[name, label] for name, label in rows]
    metadata = {"labels": labels if all(label is not None for _, label in rows) else None}
    _atomic_write(os.path.join(dest, "dataset.json"), json.dumps(metadata).encode("utf-8"))


@click.group()
def cmdline() -> None:
    """Parallel folder-based ImageNet preprocessing."""


@cmdline.command()
@click.option("--source", type=str, required=True)
@click.option("--dest", type=str, required=True)
@click.option("--max-images", type=int)
@click.option("--transform", type=click.Choice(["center-crop", "center-crop-wide", "center-crop-dhariwal"]))
@click.option("--resolution", type=parse_tuple)
@click.option("--workers", type=click.IntRange(min=1), default=max(1, (os.cpu_count() or 8) // 2), show_default=True)
@click.option("--resume", is_flag=True, help="Reuse non-empty outputs and skip completed files")
def convert(source: str, dest: str, max_images: Optional[int], transform: Optional[str], resolution: Optional[tuple[int, int]], workers: int, resume: bool) -> None:
    """Crop/resize a folder dataset using multiple CPU processes."""
    if not os.path.isdir(source):
        raise click.ClickException("Parallel convert currently requires a source directory")
    dest = _prepare_dest(dest, resume)
    entries = _discover_folder(source, max_images)
    tasks = ((idx, path, label) for idx, (path, label) in enumerate(entries))
    rows: list[tuple[str, Optional[int]]] = []
    expected_size = resolution

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_convert_worker,
        initargs=(dest, transform, resolution, resume),
    ) as pool:
        results = pool.map(_convert_one, tasks, chunksize=1)
        for archive_name, label, size in tqdm(results, total=len(entries), desc="convert"):
            if expected_size is None:
                expected_size = size
            elif size != expected_size:
                raise click.ClickException(f"Mismatched image size: expected {expected_size}, got {size}")
            rows.append((archive_name, label))

    _write_metadata(dest, rows)
    click.echo(f"Wrote {len(rows)} images to {dest}")


@cmdline.command()
@click.option("--source", type=str, required=True)
@click.option("--dest", type=str, required=True)
@click.option("--model-url", type=str, default="stabilityai/sd-vae-ft-mse", show_default=True)
@click.option("--gpus", type=str, required=True, help="Comma-separated physical CUDA device IDs, e.g. 0,1,2,3")
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True, help="VAE batch size per GPU")
@click.option("--max-images", type=int)
@click.option("--resume", is_flag=True, help="Reuse non-empty outputs and skip completed files")
def encode(source: str, dest: str, model_url: str, gpus: str, batch_size: int, max_images: Optional[int], resume: bool) -> None:
    """Encode a folder dataset with one persistent worker per GPU."""
    if not os.path.isdir(source):
        raise click.ClickException("Parallel encode currently requires a source directory")
    gpu_ids = [item.strip() for item in gpus.split(",") if item.strip()]
    if not gpu_ids or any(not item.isdigit() for item in gpu_ids):
        raise click.ClickException("--gpus must be a comma-separated list of integer CUDA device IDs")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise click.ClickException("--gpus contains duplicate device IDs")
    dest = _prepare_dest(dest, resume)
    entries = _discover_folder(source, max_images)
    total_batches = (len(entries) + batch_size - 1) // batch_size

    context = mp.get_context("spawn")
    device_queue = context.Queue()
    for gpu_id in gpu_ids:
        device_queue.put(gpu_id)

    rows: list[tuple[str, Optional[int]]] = []
    with ProcessPoolExecutor(
        max_workers=len(gpu_ids),
        mp_context=context,
        initializer=_init_encode_worker,
        initargs=(model_url, dest, resume, device_queue),
    ) as pool:
        results = pool.map(_encode_batch, _batches(entries, batch_size), chunksize=1)
        for batch_rows in tqdm(results, total=total_batches, desc="encode"):
            rows.extend(batch_rows)

    _write_metadata(dest, rows)
    click.echo(f"Wrote {len(rows)} latent moments to {dest}")


if __name__ == "__main__":
    cmdline()
