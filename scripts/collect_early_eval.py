#!/usr/bin/env python3
"""Collect early-checkpoint mechanism JSON files into CSV tables."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List


def step_from_name(path: Path) -> int:
    match = re.search(r"(\d{7})", path.stem)
    return int(match.group(1)) if match else -1


def write_rows(path: Path, rows: Iterable[Dict[str, object]]):
    rows = list(rows)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {path}")


def collect_q_scaffold(run_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((run_dir / "q_scaffold_probe").glob("q_scaffold_probe_*.json")):
        with open(path) as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        step = int(meta.get("step", step_from_name(path)))
        for timestep, layer_map in payload.get("results", {}).items():
            for layer, metrics in layer_map.items():
                row = {
                    "run": run_dir.name,
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                    "model": meta.get("model"),
                    "repa_loss": meta.get("repa_loss"),
                    "distill_coeff": meta.get("distill_coeff"),
                    "kv_replace_mode": meta.get("kv_replace_mode"),
                    "encoder_patch_shuffle": meta.get("encoder_patch_shuffle"),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def collect_spatial(run_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((run_dir / "spatial_metrics").glob("spatial_metrics_*.json")):
        with open(path) as f:
            payload = json.load(f)
        step = step_from_name(path)
        for timestep, layer_map in payload.items():
            if not isinstance(layer_map, dict):
                continue
            for layer, metrics in layer_map.items():
                if not isinstance(metrics, dict):
                    continue
                row = {
                    "run": run_dir.name,
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                }
                row.update(metrics)
                rows.append(row)
    return rows


def collect_teacher_spatial_alignment(run_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted(
        (run_dir / "teacher_spatial_alignment").glob(
            "teacher_spatial_alignment_*.json"
        )
    ):
        with open(path) as f:
            payload = json.load(f)
        meta = payload.get("meta", {})
        step = int(meta.get("step", step_from_name(path)))
        for timestep, layer_map in payload.get("results", {}).items():
            for layer, metrics in layer_map.items():
                row = {
                    "run": run_dir.name,
                    "step": step,
                    "timestep": timestep,
                    "layer": layer,
                    "model": meta.get("model"),
                    "teacher_enc_type": meta.get("teacher_enc_type"),
                    "teacher_patch_shuffle": meta.get("teacher_patch_shuffle"),
                    "repa_loss": meta.get("repa_loss"),
                    "distill_coeff": meta.get("distill_coeff"),
                    "kv_replace_mode": meta.get("kv_replace_mode"),
                    "encoder_patch_shuffle": meta.get("encoder_patch_shuffle"),
                }
                row.update(metrics)
                rows.append(row)
    return rows


def collect_linear_probe(run_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for path in sorted((run_dir / "linear_probe").glob("linear_probe_*.json")):
        with open(path) as f:
            payload = json.load(f)
        args = payload.get("args", {})
        best = payload.get("best", {})
        metrics = best.get("metrics", {})
        rows.append({
            "run": run_dir.name,
            "step": step_from_name(path),
            "layer_depth": args.get("layer_depth"),
            "timestep": args.get("timestep"),
            "pool": args.get("pool"),
            "best_epoch": best.get("epoch"),
            "val_top1": best.get("top1"),
            "val_top5": metrics.get("top5"),
            "val_loss": metrics.get("loss"),
            "num_samples": metrics.get("num_samples"),
        })
    return rows


def main(args):
    run_dir = Path(args.run_dir).resolve()
    out_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "early_eval"
    write_rows(out_dir / "q_scaffold_probe.csv", collect_q_scaffold(run_dir))
    write_rows(out_dir / "spatial_metrics.csv", collect_spatial(run_dir))
    write_rows(
        out_dir / "teacher_spatial_alignment.csv",
        collect_teacher_spatial_alignment(run_dir),
    )
    write_rows(out_dir / "linear_probe.csv", collect_linear_probe(run_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect early checkpoint eval JSON")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    main(parser.parse_args())
