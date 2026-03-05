"""
generate_comparison_grid_v2.py
-------------------------------
Generate a publication-quality comparison grid figure.

Layout (same as Figure 8 style):
  - Rows  : 2 methods × 2 (for each label row pair)
  - Cols  : 3 ckpt iterations per (label, seed) group
  - Groups: 6 (label, seed) combinations → displayed as column groups

You provide:
  - 6 checkpoints: 3 for method A + 3 for method B
  - 6 class labels and 6 seeds (paired 1:1)

Usage example:

    CUDA_VISIBLE_DEVICES=0 python generate_comparison_grid_v2.py \
        --method-a-label "SiT-XL/2" \
        --method-b-label "SiT-XL/2+REPA" \
        --method-a-ckpts ckpt_a_100k.pt ckpt_a_200k.pt ckpt_a_400k.pt \
        --method-b-ckpts ckpt_b_100k.pt ckpt_b_200k.pt ckpt_b_400k.pt \
        --ckpt-labels 100K 200K 400K \
        --class-labels 207 88 2 400 849 325 \
        --seeds 0 1 2 42 72 142 \
        --cfg-scale 1.0 \
        --num-steps 250 \
        --mode ode \
        --out comparison_grid.png
"""

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

# ──────────────────────────────────────────────────────────────
# SIT codebase imports
# ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.sit import SiT_models
from models.autoencoder import VAE_F8D4
from samplers import euler_sampler, euler_maruyama_sampler


# ══════════════════════════════════════════════════════════════
#  Model helpers
# ══════════════════════════════════════════════════════════════

def load_vae(device):
    """Load SD-VAE and latent normalisation stats."""
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
    bias  = stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return vae, scale, bias


def load_model(ckpt_path, model_name, device, resolution=256):
    """Load a SiT model from checkpoint (EMA weights, strict=False)."""
    print(f"  Loading {ckpt_path} …")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    ckpt_args = ckpt.get("args", None)
    qk_norm = False
    if ckpt_args is not None and hasattr(ckpt_args, "qk_norm"):
        qk_norm = ckpt_args.qk_norm

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
    filtered   = {k: v for k, v in state_dict.items() if "kv_proj" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if missing:
        print(f"    [info] {len(missing)} missing keys")
    if unexpected:
        print(f"    [warn] {len(unexpected)} unexpected keys")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════
#  Sampling
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_image(model, vae, latents_scale, latents_bias,
                   class_label: int, seed: int,
                   cfg_scale: float, num_steps: int,
                   mode: str, path_type: str,
                   resolution: int, device):
    """
    Generate a single image with a fixed seed and class label.

    Returns: PIL.Image (RGB, resolution × resolution)
    """
    latent_size = resolution // 8
    torch.manual_seed(seed)

    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    y = torch.full((1,), class_label, dtype=torch.long, device=device)

    sampling_kwargs = dict(
        model=model,
        latents=z,
        y=y,
        num_steps=num_steps,
        heun=False,
        cfg_scale=cfg_scale,
        guidance_low=0.0,
        guidance_high=1.0,
        path_type=path_type,
    )

    if mode == "sde":
        samples = euler_maruyama_sampler(**sampling_kwargs).to(torch.float32)
    else:
        samples = euler_sampler(**sampling_kwargs).to(torch.float32)

    samples = vae.decode(samples / latents_scale + latents_bias).sample
    samples = (samples + 1) / 2.0
    samples = torch.clamp(255.0 * samples, 0, 255) \
                   .permute(0, 2, 3, 1) \
                   .byte() \
                   .cpu() \
                   .numpy()

    return Image.fromarray(samples[0])


# ══════════════════════════════════════════════════════════════
#  Grid layout & plotting
# ══════════════════════════════════════════════════════════════

def make_grid_figure(
    images,          # images[method_idx][ckpt_idx][group_idx] = PIL.Image
    method_labels,   # e.g. ["SiT-XL/2", "SiT-XL/2+REPA"]
    ckpt_labels,     # e.g. ["100K", "200K", "400K"]
    group_titles,    # per-(label,seed) group titles
    save_path,
    groups_per_row=3,
    dpi=200,
    cell_size=2.2,
):
    """
    Draw the comparison grid matching Figure 8 style.

    When there are N groups total and groups_per_row=3, we create
    ceil(N/3) "super-rows", each containing:
      - method_A row with groups_per_row × n_ckpts images
      - method_B row with groups_per_row × n_ckpts images

    Layout for groups_per_row=3 and 6 groups:
      ┌─────────────── Super-row 0 ────────────────────────────┐
      │         [group 0]          [group 1]         [group 2] │
      │       100K 200K 400K    100K 200K 400K    100K 200K 400K│
      │ A      img  img  img     img  img  img     img  img  img│
      │ B      img  img  img     img  img  img     img  img  img│
      ├─────────────── Super-row 1 ────────────────────────────┤
      │         [group 3]          [group 4]         [group 5] │
      │       100K 200K 400K    100K 200K 400K    100K 200K 400K│
      │ A      img  img  img     img  img  img     img  img  img│
      │ B      img  img  img     img  img  img     img  img  img│
      └───────────────────────────────────────────────────────-─┘
    """
    import math

    n_ckpts   = len(ckpt_labels)
    n_groups  = len(group_titles)
    n_methods = len(method_labels)

    n_super_rows = math.ceil(n_groups / groups_per_row)

    # Total grid dimensions
    n_cols_total = groups_per_row * n_ckpts
    n_rows_total = n_super_rows * n_methods

    fig_w = cell_size * n_cols_total + 1.2
    fig_h = cell_size * n_rows_total + 0.9 * n_super_rows

    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = gridspec.GridSpec(
        n_rows_total, n_cols_total,
        left=0.06, right=0.99, top=0.93, bottom=0.01,
        wspace=0.04, hspace=0.08,
    )

    for super_row in range(n_super_rows):
        grp_start = super_row * groups_per_row
        grp_end   = min(grp_start + groups_per_row, n_groups)

        for m_idx, m_label in enumerate(method_labels):
            row = super_row * n_methods + m_idx

            for local_grp, grp_idx in enumerate(range(grp_start, grp_end)):
                for ckpt_idx in range(n_ckpts):
                    col = local_grp * n_ckpts + ckpt_idx
                    ax = fig.add_subplot(gs[row, col])

                    img = images[m_idx][ckpt_idx][grp_idx]
                    ax.imshow(np.array(img))
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # Thin separator between groups
                    lw_left = 2.0 if ckpt_idx == 0 and local_grp > 0 else 0.5
                    for spine in ax.spines.values():
                        spine.set_linewidth(0.5)
                        spine.set_color("#cccccc")
                    ax.spines["left"].set_linewidth(lw_left)
                    ax.spines["left"].set_color(
                        "#888888" if lw_left > 1 else "#cccccc"
                    )

                    # Row label on leftmost column
                    if ckpt_idx == 0 and local_grp == 0:
                        ax.set_ylabel(
                            m_label,
                            fontsize=7.5,
                            fontweight="bold",
                            rotation=90,
                            labelpad=4,
                            va="center",
                        )

                    # Ckpt label on first method row of each super-row
                    if m_idx == 0:
                        ax.set_title(ckpt_labels[ckpt_idx], fontsize=7, pad=2)

            # Group title above first method row
            if m_idx == 0:
                for local_grp, grp_idx in enumerate(range(grp_start, grp_end)):
                    col_start = local_grp * n_ckpts
                    col_end   = col_start + n_ckpts - 1
                    ax_span = fig.add_subplot(
                        gs[row, col_start : col_end + 1]
                    )
                    ax_span.set_visible(False)
                    ax_span.text(
                        0.5, 1.25, group_titles[grp_idx],
                        ha="center", va="bottom",
                        fontsize=8, fontweight="bold",
                        transform=ax_span.transAxes,
                    )

    # Add "Training Iteration" arrow header at top
    fig.text(
        0.52, 0.98, "Training Iteration →",
        ha="center", va="top",
        fontsize=9, fontweight="bold",
    )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ══════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════

IMAGENET_NAMES = {
    0: "tench", 1: "goldfish", 2: "great white shark", 7: "hen",
    15: "robin", 80: "ruffed grouse", 81: "ptarmigan",
    85: "quail", 86: "partridge", 88: "macaw",
    89: "sulphur-crested cockatoo", 101: "tusker",
    127: "white stork", 130: "flamingo",
    207: "golden retriever", 208: "Labrador retriever",
    231: "Saluki", 234: "Rottweiler", 235: "Doberman",
    245: "French bulldog", 246: "miniature poodle",
    250: "Siberian husky", 263: "Pembroke",
    267: "standard poodle", 281: "tabby cat",
    325: "fire truck", 340: "zebra",
    344: "hippopotamus", 346: "water buffalo",
    347: "bison", 385: "Indian elephant",
    386: "African elephant", 388: "giant panda",
    400: "hot air balloon", 505: "coffee mug",
    555: "fire engine", 569: "fire truck",
    717: "pickup truck", 724: "sports car",
    849: "cheeseburger", 933: "cheeseburger",
    968: "cup", 985: "daisy",
}


def imagenet_classname(class_id: int) -> str:
    return IMAGENET_NAMES.get(class_id, f"cls{class_id}")


# ══════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training-iteration comparison grid (Figure 8 style).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Checkpoints ──────────────────────────────────────────
    parser.add_argument("--method-a-ckpts", nargs=3, required=True,
                        metavar="CKPT",
                        help="3 checkpoint paths for method A (in iteration order)")
    parser.add_argument("--method-b-ckpts", nargs=3, required=True,
                        metavar="CKPT",
                        help="3 checkpoint paths for method B (in iteration order)")
    parser.add_argument("--method-a-label", type=str, default="SiT-XL/2",
                        help="Row label for method A")
    parser.add_argument("--method-b-label", type=str, default="SiT-XL/2+REPA",
                        help="Row label for method B")
    parser.add_argument("--ckpt-labels", nargs=3, default=["100K", "200K", "400K"],
                        metavar="LABEL",
                        help="Display labels for each checkpoint column")

    # ── Class labels and seeds (paired 1:1) ───────────────────
    parser.add_argument("--class-labels", type=int, nargs="+", required=True,
                        metavar="CLS",
                        help="ImageNet class IDs, e.g. 207 88 2 400 849 325")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        metavar="S",
                        help="Seeds paired 1:1 with class-labels, e.g. 0 1 2 42 72 142")

    # ── Grid layout ──────────────────────────────────────────
    parser.add_argument("--groups-per-row", type=int, default=3,
                        help="Number of (label,seed) groups per visual row "
                             "(default 3, matching Figure 8)")

    # ── Sampling ─────────────────────────────────────────────
    parser.add_argument("--cfg-scale", type=float, default=4.0,
                        help="Classifier-free guidance scale (paper: 4.0)")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--mode", type=str, default="ode",
                        choices=["ode", "sde"])
    parser.add_argument("--path-type", type=str, default="linear",
                        choices=["linear", "cosine"])
    parser.add_argument("--resolution", type=int, default=256,
                        choices=[256, 512])
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()))

    # ── Output ───────────────────────────────────────────────
    parser.add_argument("--out", type=str, default="comparison_grid_v2.png",
                        help="Output image path (PNG or PDF)")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--cell-size", type=float, default=2.2,
                        help="Cell size in inches for each image cell")

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate: class-labels and seeds must have the same length
    if len(args.class_labels) != len(args.seeds):
        raise ValueError(
            f"Number of class-labels ({len(args.class_labels)}) must equal "
            f"number of seeds ({len(args.seeds)}). They are paired 1:1."
        )

    combos = list(zip(args.class_labels, args.seeds))
    n_groups = len(combos)
    n_ckpts  = 3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    print(f"\nClass labels: {args.class_labels}")
    print(f"Seeds:        {args.seeds}")
    print(f"Groups (label, seed): {combos}")
    print(f"Total images to generate: "
          f"{n_groups} × {n_ckpts} ckpts × 2 methods = "
          f"{n_groups * n_ckpts * 2}")

    # ------------------------------------------------------------------
    # Load VAE
    # ------------------------------------------------------------------
    print("\nLoading VAE …")
    vae, latents_scale, latents_bias = load_vae(device)

    # ------------------------------------------------------------------
    # Load all 6 models
    # ------------------------------------------------------------------
    print("\nLoading method A models …")
    models_a = [
        load_model(p, args.model, device, args.resolution)
        for p in args.method_a_ckpts
    ]
    print("\nLoading method B models …")
    models_b = [
        load_model(p, args.model, device, args.resolution)
        for p in args.method_b_ckpts
    ]

    # ------------------------------------------------------------------
    # Generate images
    #   images[method_idx][ckpt_idx][group_idx] = PIL.Image
    # ------------------------------------------------------------------
    all_models = [models_a, models_b]
    method_labels_list = [args.method_a_label, args.method_b_label]

    # images[m_idx][ck_idx][grp_idx]
    images = [[[] for _ in range(n_ckpts)] for _ in range(2)]

    total_gens = n_groups * n_ckpts * 2
    done = 0
    for m_idx, (models, m_label) in enumerate(
            zip(all_models, method_labels_list)):
        for ck_idx, (model, ck_lbl) in enumerate(
                zip(models, args.ckpt_labels)):
            for grp_idx, (cl, seed) in enumerate(combos):
                done += 1
                cls_name = imagenet_classname(cl)
                print(f"  [{done}/{total_gens}] "
                      f"{m_label} / {ck_lbl} "
                      f"label={cl}({cls_name}) seed={seed} …")
                img = generate_image(
                    model=model,
                    vae=vae,
                    latents_scale=latents_scale,
                    latents_bias=latents_bias,
                    class_label=cl,
                    seed=seed,
                    cfg_scale=args.cfg_scale,
                    num_steps=args.num_steps,
                    mode=args.mode,
                    path_type=args.path_type,
                    resolution=args.resolution,
                    device=device,
                )
                images[m_idx][ck_idx].append(img)
    print()

    # ------------------------------------------------------------------
    # Build group titles
    # ------------------------------------------------------------------
    group_titles = []
    for cl, seed in combos:
        cls_name = imagenet_classname(cl)
        group_titles.append(f"{cls_name} (cls={cl}, seed={seed})")

    # ------------------------------------------------------------------
    # Draw and save the grid figure
    # ------------------------------------------------------------------
    print(f"Drawing grid figure …")
    make_grid_figure(
        images=images,
        method_labels=method_labels_list,
        ckpt_labels=args.ckpt_labels,
        group_titles=group_titles,
        save_path=args.out,
        groups_per_row=args.groups_per_row,
        dpi=args.dpi,
        cell_size=args.cell_size,
    )

    print("\nDone! 🎉")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
