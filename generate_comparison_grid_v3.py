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

    CUDA_VISIBLE_DEVICES=0 python generate_comparison_grid_v3.py \
    --method-a-label "iREPA" \
    --method-b-label "AttnScaf" \
    --method-a-ckpts /workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt /workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0200000.pt /workspace/SIT/iREPA-collections/0400000.pt \
    --method-b-ckpts /workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt /workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0200000.pt /workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0400000.pt \
    --ckpt-labels 100K 200K 400K \
    --class-labels 335 31 511 200 417 127 \
    --seeds 1 142 43 45 2 67 \
    --cfg-scale 1.0 --num-steps 250 --mode ode \
    --out wocfg_compare.pdf \
    --fontsize 18
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
#  Grid layout & plotting (ECCV publication style)
# ══════════════════════════════════════════════════════════════

def make_grid_figure(
    images,          # images[method_idx][ckpt_idx][group_idx] = PIL.Image
    method_labels,   # e.g. ["SiT-XL/2", "SiT-XL/2+REPA"]
    ckpt_labels,     # e.g. ["100K", "200K", "400K"]
    group_titles,    # per-(label,seed) group titles
    save_path,
    groups_per_row=3,
    dpi=300,
    fig_width=13.5,    # Large figure; shrink in LaTeX with \includegraphics[width=\textwidth]
    font_family="STIXGeneral",
    fontsize=13,
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
    from matplotlib.patches import FancyArrowPatch

    # --- Font configuration (ECCV style) ---
    plt.rcParams.update({
        "font.family": font_family,
    })

    n_ckpts   = len(ckpt_labels)
    n_groups  = len(group_titles)
    n_methods = len(method_labels)

    n_super_rows = math.ceil(n_groups / groups_per_row)

    # Total grid dimensions
    n_cols_total = groups_per_row * n_ckpts

    # Derive cell_w from fig_width
    row_label_margin = 0.6
    cell_w = (fig_width - row_label_margin) / n_cols_total
    cell_h = cell_w  # keep cells square

    # Build height_ratios: insert gap rows between super-rows
    # e.g. for 2 super-rows × 2 methods:  [1, 1, gap, 1, 1]
    super_row_gap = 0.03  # relative to one cell row
    height_ratios = []
    for sr in range(n_super_rows):
        if sr > 0:
            height_ratios.append(super_row_gap)  # gap row
        height_ratios.extend([1.0] * n_methods)
    n_gs_rows = len(height_ratios)

    fig_w = fig_width
    n_real_rows = n_super_rows * n_methods
    gs_top = 0.95
    gs_bottom = 0.01
    content_h = cell_h * n_real_rows + cell_h * super_row_gap * max(0, n_super_rows - 1)
    fig_h = content_h / (gs_top - gs_bottom)

    fig = plt.figure(figsize=(fig_w, fig_h))

    # Build width_ratios with small gaps between groups
    gap_ratio = 0.08  # small gap between column groups
    width_ratios = []
    for grp_i in range(groups_per_row):
        if grp_i > 0:
            width_ratios.append(gap_ratio)
        width_ratios.extend([1.0] * n_ckpts)
    n_gs_cols = len(width_ratios)

    gs = gridspec.GridSpec(
        n_gs_rows, n_gs_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        left=0.06, right=0.99, top=gs_top, bottom=gs_bottom,
        wspace=0.02, hspace=0.01,
    )

    def gs_col(local_grp, ckpt_idx):
        """Map (local_grp, ckpt_idx) to gridspec column index."""
        return local_grp * (n_ckpts + 1) + ckpt_idx if local_grp > 0 \
            else ckpt_idx

    def gs_row(super_row, m_idx):
        """Map (super_row, method_idx) to gridspec row index, accounting for gap rows."""
        return super_row * (n_methods + 1) + m_idx if super_row > 0 \
            else m_idx

    for super_row in range(n_super_rows):
        grp_start = super_row * groups_per_row
        grp_end   = min(grp_start + groups_per_row, n_groups)

        for m_idx, m_label in enumerate(method_labels):
            row = gs_row(super_row, m_idx)

            for local_grp, grp_idx in enumerate(range(grp_start, grp_end)):
                for ckpt_idx in range(n_ckpts):
                    col = gs_col(local_grp, ckpt_idx)
                    ax = fig.add_subplot(gs[row, col])

                    # Track axes for arrow positioning
                    if super_row == 0 and m_idx == 0 and local_grp == 0:
                        if ckpt_idx == 0:
                            _arrow_ax_left = ax
                        if ckpt_idx == n_ckpts - 1:
                            _arrow_ax_right = ax

                    img = images[m_idx][ckpt_idx][grp_idx]
                    ax.imshow(np.array(img))
                    ax.set_xticks([])
                    ax.set_yticks([])

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    # Ckpt label only on the very first super-row
                    if m_idx == 0 and super_row == 0:
                        ax.set_title(
                            ckpt_labels[ckpt_idx],
                            fontsize=fontsize + 2, fontweight="normal",
                            pad=2,
                        )

            # Group title above first method row (skip group 0 — arrow goes there)
            if m_idx == 0:
                for local_grp, grp_idx in enumerate(range(grp_start, grp_end)):
                    if super_row == 0 and local_grp == 0:
                        continue  # arrow replaces title for first group
                    col_start = gs_col(local_grp, 0)
                    col_end   = gs_col(local_grp, n_ckpts - 1)
                    ax_span = fig.add_subplot(
                        gs[row, col_start : col_end + 1]
                    )
                    ax_span.set_visible(False)
                    ax_span.text(
                        0.5, 1.30, group_titles[grp_idx],
                        ha="center", va="bottom",
                        fontsize=fontsize, fontweight="bold",
                        transform=ax_span.transAxes,
                    )

    # ---- Row labels via fig.text() ----
    gs_left = gs.left
    for super_row in range(n_super_rows):
        for m_idx, m_label in enumerate(method_labels):
            row = gs_row(super_row, m_idx)
            # Vertical center of this gridspec row
            cum_before = sum(height_ratios[:row])
            cum_after  = sum(height_ratios[:row + 1])
            total_h    = sum(height_ratios)
            frac_top = cum_before / total_h
            frac_bot = cum_after / total_h
            y_top = gs.top - frac_top * (gs.top - gs.bottom)
            y_bot = gs.top - frac_bot * (gs.top - gs.bottom)
            y_center = (y_top + y_bot) / 2
            fig.text(
                gs_left - 0.005, y_center, m_label,
                ha="right", va="center",
                fontsize=fontsize, fontweight="normal",
                rotation=90,
            )

    # ---- "Training Iteration" arrow spanning the first column group ----
    fig.canvas.draw()  # force layout computation

    bbox_l = _arrow_ax_left.get_position()
    bbox_r = _arrow_ax_right.get_position()

    arrow_y = bbox_l.y1 + 0.06  # just above the ckpt labels
    arrow_x0 = bbox_l.x0
    arrow_x1 = bbox_r.x1

    # Draw arrow
    arrow = FancyArrowPatch(
        (arrow_x0, arrow_y), (arrow_x1, arrow_y),
        transform=fig.transFigure,
        arrowstyle="->,head_width=2.5,head_length=2.5",
        color="black", linewidth=0.8,
        clip_on=False,
    )
    fig.patches.append(arrow)

    # "Training Iteration" text centered on the arrow
    fig.text(
        (arrow_x0 + arrow_x1) / 2, arrow_y + 0.01,
        "Training Iteration",
        ha="center", va="bottom",
        fontsize=fontsize + 1, fontweight="normal",
    )

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
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

    # ── Output & figure style ────────────────────────────────
    parser.add_argument("--out", type=str, default="comparison_grid_v2.pdf",
                        help="Output file path (.pdf or .png)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--fig-width", type=float, default=13.5,
                        help="Total figure width in inches. Default 13.5 (large, "
                             "rescale in LaTeX). ECCV textwidth=6.7.")
    parser.add_argument("--font", default="STIXGeneral",
                        help="Font name, e.g. 'STIXGeneral', 'DejaVu Sans', 'Arial'")
    parser.add_argument("--fontsize", type=float, default=9,
                        help="Base font size in pt (default 9). ECCV: try 8-9.")

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
        fig_width=args.fig_width,
        font_family=args.font,
        fontsize=args.fontsize,
    )

    print("\nDone! 🎉")
    print(f"  Output: {args.out}")


if __name__ == "__main__":
    main()
