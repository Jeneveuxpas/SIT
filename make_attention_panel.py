"""
make_attention_panel.py
=======================
Assembles a 2-row × 8-column publication figure from attention heatmaps,
matching the reference layout:

    Row 0:  [img_A | method1 | method2 | method3]  [img_B | method1 | method2 | method3]
    Row 1:  [img_C | method1 | method2 | method3]  [img_D | method1 | method2 | method3]

The 8 columns = 4 panels × (1 original + 3 methods).
There are 3 methods ("Vanilla", "iREPA", "Ours") — configurable via --labels.

Two modes
---------
A) --from-images  (no model needed)
   Pass pre-computed heatmap arrays (.npy) and original images directly.
   Useful when you have already run visualize_attention.py and saved results.

B) --from-ckpts   (live inference)
   Load checkpoints, run forward passes, compute cosine-similarity maps on
   the fly, then assemble the panel.  This reuses helpers from
   visualize_attention.py.

Layout
------
  Total cols : 8   (4 groups × [orig + 3 methods])
  Total rows : 2
  Extra right: colorbar
  Row labels (y-axis): user-supplied via --row-labels
  Group labels (top): user-supplied via --group-labels  (one per group, placed
                       above the 2nd column of each group, i.e. above method1)

Usage example (from-ckpts mode)
---------------------------------
  CUDA_VISIBLE_DEVICES=0 python make_attention_panel.py \
    --images images/bird1.jpg images/bird1.jpg images/sleep.jpg images/sleep.jpg \
    --queries "2,12"  "7,6"  "2,10"  "6,7" \
    --ckpts \
        "Vanilla:/workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt" \
        "iREPA:/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
        "AttnScaf:/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
    --model SiT-XL/2 \
    --layer 8 \
    --timestep 0.1 \
    --out output/panel.pdf \
    --row-label $'Diffusion attention output\n(SiT-XL/2 Layer 8)' \
    --labels "Vanilla" "iREPA" "AttnScaf" \
    --row-label-x 0.7  \
    --fig-width 6.7 \
    --fontsize 9 \
    --font "STIXGeneral" 

  The 4 images map to:
    panel[0,0..3] ← images[0] + queries[0]   (row 0, group 0)
    panel[0,4..7] ← images[1] + queries[1]   (row 0, group 1)
    panel[1,0..3] ← images[2] + queries[2]   (row 1, group 0)
    panel[1,4..7] ← images[3] + queries[3]   (row 1, group 1)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Reuse helpers from visualize_attention.py
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualize_attention import (
    load_image,
    load_vae,
    load_sit_model,
    encode_to_latent,
    extract_from_layer,
    resolve_query,
)


# ===========================================================================
# Heatmap computation
# ===========================================================================

def compute_heatmap(data, q_idx, grid_size, viz_mode="attn_output", label=""):
    """
    Convert raw extracted tensor into a (grid_size, grid_size) heatmap.

    data     - Tensor from extract_from_layer:
                 attn_weights → (1, heads, N, N)
                 attn_output / feature_sim → (1, N, C)
    q_idx    - flat query token index
    """
    if viz_mode == "attn_weights":
        attn_map = data[0].mean(dim=0).numpy()   # (N, N)
        hmap = attn_map[q_idx]                   # (N,)
    else:
        features = data[0]                        # (N, C)
        query_feat = features[q_idx]
        hmap = F.cosine_similarity(
            query_feat.unsqueeze(0), features, dim=-1
        ).numpy()
    # Debug: print actual value range
    print(f"  [{label or 'method'}] cosine sim: "
          f"min={hmap.min():.4f}  max={hmap.max():.4f}  "
          f"mean={hmap.mean():.4f}  std={hmap.std():.6f}")
    return hmap.reshape(grid_size, grid_size)


# ===========================================================================
# Panel assembly
# ===========================================================================

def assemble_panel(
    cell_data,
    method_labels,
    row_label,
    group_labels,
    save_path,
    cmap="viridis",
    vmin=None, vmax=None,
    row_label_x=0.5,
    font_family="sans-serif",
    fontsize=9,        # base font size for column/row labels (pt in the saved PDF)
    fig_width=None,    # total figure width in inches; None = auto from cell_w
):
    """
    Draw the 2 × 8 panel.

    cell_data  is ordered [top-left, top-right, bottom-left, bottom-right]:
        cell 0 → row 0, group 0  (cols 0-3)
        cell 1 → row 0, group 1  (cols 4-7)
        cell 2 → row 1, group 0  (cols 0-3)
        cell 3 → row 1, group 1  (cols 4-7)
    """
    # --- Font configuration --------------------------------------------------
    # font_family can be any font name, e.g. "Arial", "Times New Roman",
    # "STIXGeneral", "DejaVu Sans", "Helvetica", etc.
    plt.rcParams.update({
        "font.family": font_family,
    })
    N_ROWS   = 2
    N_GROUPS = 2
    N_METHODS = len(method_labels)          # 3
    COLS_PER_GROUP = 1 + N_METHODS          # 4
    N_COLS   = N_GROUPS * COLS_PER_GROUP    # 8

    # --- Global normalization across all heatmaps ---------------------------
    # All heatmaps share the same vmin/vmax so that colors represent true
    # cosine similarity values.  This means Vanilla (spatially uniform:
    # all values near 1.0) will appear uniformly yellow — which is the
    # correct scientific result showing its lack of spatial structure.
    all_vals = np.concatenate([
        c["heatmaps"][i].ravel()
        for c in cell_data
        for i in range(N_METHODS)
    ])
    if vmin is None:
        vmin = float(all_vals.min())
    if vmax is None:
        vmax = float(all_vals.max())

    # Figure sizing -----------------------------------------------------------
    # If fig_width is given, derive cell_w from it; otherwise use default 2.2"
    # ECCV textwidth ≈ 6.7". Designing the figure at that width means font sizes
    # are already at print scale and LaTeX won't shrink them.
    if fig_width is not None:
        # subtract fixed widths (gap + colorbar + row-label margin)
        fixed_w = 0.15 + 0.35 + 0.9
        cell_w = (fig_width - fixed_w) / N_COLS
    else:
        cell_w = 2.2
    cell_h = cell_w          # keep cells square
    gap_w  = 0.15
    cbar_w = 0.35

    fig_w = N_COLS * cell_w + gap_w + cbar_w + 0.9
    fig_h = N_ROWS * cell_h + fontsize / 72 * 4  # headroom for column titles

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: 2 rows × (N_COLS + 1 cbar)
    gap_ratio = 0.18   # relative to one cell

    width_ratios = (
        [1.0] * COLS_PER_GROUP
        + [gap_ratio]
        + [1.0] * COLS_PER_GROUP
        + [0.12]   # colorbar
    )
    # Total cols in gs: 4 + 1 + 4 + 1 = 10
    gs = gridspec.GridSpec(
        N_ROWS, N_COLS + 2,   # +1 gap, +1 colorbar
        width_ratios=width_ratios,
        wspace=0.04,
        hspace=0.08,
        left=0.14, right=0.97,
        top=0.90,  bottom=0.03,
    )

    # Column mapping: group 0 → gs cols 0-3; gap → gs col 4; group 1 → gs cols 5-8; cbar → gs col 9
    def gs_col(group_idx, local_col):
        if group_idx == 0:
            return local_col                # 0-3
        else:
            return COLS_PER_GROUP + 1 + local_col   # 5-8

    # Scale marker size with cell width so stars don't dominate at small fig sizes
    _ms_scale = cell_w / 2.2   # 1.0 at default cell_w=2.2", smaller when fig_width given
    MARKER_KWARGS_PIX  = dict(marker="*", color="red", markersize=max(18 * _ms_scale, 4),
                               markeredgecolor="darkred", markeredgewidth=max(0.7 * _ms_scale, 0.3),
                               linestyle="None")
    MARKER_KWARGS_GRID = dict(marker="*", color="red", markersize=max(14 * _ms_scale, 4),
                               markeredgecolor="darkred", markeredgewidth=max(0.5 * _ms_scale, 0.2),
                               linestyle="None")

    all_ims = []

    for row in range(N_ROWS):
        for grp in range(N_GROUPS):
            cell_idx = row * N_GROUPS + grp
            cell = cell_data[cell_idx]

            orig_img = cell["orig"]
            heatmaps = cell["heatmaps"]
            qx_pix, qy_pix   = cell["q_pix"]
            qx_grid, qy_grid  = cell["q_grid"]

            # ---- Col 0 of group: original image ----------------------------
            ax0 = fig.add_subplot(gs[row, gs_col(grp, 0)])
            ax0.imshow(orig_img)
            ax0.plot(qx_pix, qy_pix, **MARKER_KWARGS_PIX)
            ax0.set_xticks([])
            ax0.set_yticks([])
            for spine in ax0.spines.values():
                spine.set_visible(False)

            # Row label: drawn once per row as rotated fig.text, to the left of group 0
            # (done after the loop, see below)

            # Group/column header: top row only
            if row == 0:
                ax0.set_title("Input", fontsize=fontsize, fontweight="normal", pad=3)

            # ---- Cols 1-3 of group: method heatmaps -----------------------
            for m_idx, method in enumerate(method_labels):
                gc = gs_col(grp, 1 + m_idx)
                ax = fig.add_subplot(gs[row, gc])
                # Shared global vmin/vmax — colors = true cosine similarity
                im = ax.imshow(
                    heatmaps[m_idx],
                    cmap=cmap,
                    vmin=vmin, vmax=vmax,
                    interpolation="nearest",
                    aspect="equal",
                )
                all_ims.append(im)
                ax.plot(qx_grid, qy_grid, **MARKER_KWARGS_GRID)
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)

                if row == 0:
                    ax.set_title(method, fontsize=fontsize, fontweight="normal", pad=3)

    # ---- Single shared rotated label on the far left -------------------------
    # row_label_x: fraction between 0 (figure edge) and gs.left (first subplot)
    y_center = (gs.top + gs.bottom) / 2
    x_label  = row_label_x * gs.left   # e.g. 0.5 → midpoint
    if row_label:
        fig.text(
            x_label, y_center, row_label,
            ha="center", va="center",
            fontsize=fontsize, fontweight="normal",
            rotation=90,
            multialignment="center",
        )

    # ---- Group labels above the panel (above group centers) ----------------
    if group_labels:
        for grp, glabel in enumerate(group_labels):
            # Put a super-title above the center of each group
            # x position: midpoint of gs_col(grp, 0) .. gs_col(grp, COLS_PER_GROUP-1)
            # We use fig.text with approximate x coordinates.
            # The grid spans left=0.06 to right=0.97 (minus cbar).
            data_width = 0.97 - 0.06   # approximate
            # Each group of 4 cols occupies ~ COLS_PER_GROUP / (N_COLS + gap_ratio + 0.12) of total
            total_ratio_sum = sum(width_ratios)
            grp_col_start = gs_col(grp, 0)
            grp_col_end   = gs_col(grp, COLS_PER_GROUP - 1)
            # Approximate centre in figure fraction
            left_ratio = sum(width_ratios[:grp_col_start])
            right_ratio = sum(width_ratios[:grp_col_end + 1])
            cx = 0.06 + data_width * (left_ratio + right_ratio) / 2 / total_ratio_sum
            fig.text(
                cx, 0.95, glabel,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
            )

    # ---- Shared colorbar fixed to [0, 1] with clean ticks -------------------
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    cbar_ax = fig.add_subplot(gs[:, -1])
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=[0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.set_label("Cosine Similarity", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=max(fontsize - 1, 6))

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close()
    print(f"Saved panel → {save_path}")


# ===========================================================================
# Main
# ===========================================================================

def parse_ckpt_arg(s):
    """'Label:path' or plain 'path'."""
    if ":" in s and not os.path.isabs(s.split(":")[0]):
        label, path = s.split(":", 1)
        return label.strip(), path.strip()
    path = s.strip()
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "checkpoints" and i > 0:
            return parts[i - 1], path
    return os.path.splitext(os.path.basename(path))[0], path


def main():
    parser = argparse.ArgumentParser(
        description="Build a 2-row × 8-col attention comparison panel.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # --- Panel layout inputs ------------------------------------------------
    parser.add_argument(
        "--images", nargs=4, required=True, metavar="IMG",
        help="4 image paths: [row0_grp0, row0_grp1, row1_grp0, row1_grp1]",
    )
    parser.add_argument(
        "--queries", nargs=4, required=True, metavar="Q",
        help='4 query positions (e.g. "8,8" or "center") matching --images order',
    )
    parser.add_argument(
        "--row-label",
        default="Diffusion Attn\n(SiT-XL/2 Layer 14)",
        metavar="LABEL",
        help="Single shared vertical label on the left side of the panel (use \\n for newlines)",
    )
    parser.add_argument(
        "--group-labels", nargs=2, default=None, metavar="LABEL",
        help="Optional super-titles for the 2 groups of 4 columns",
    )
    parser.add_argument(
        "--labels", nargs=3, default=["Vanilla", "iREPA", "Ours"], metavar="NAME",
        help="Names for the 3 methods (columns 2-4 within each group)",
    )
    parser.add_argument("--font", default="DejaVu Sans",
        help="Font name, e.g. 'DejaVu Sans' (default), 'Arial', 'Times New Roman', 'STIXGeneral'")
    parser.add_argument("--fontsize", type=float, default=9,
        help="Base font size in pt (default 9). For ECCV full-width figures, try 9.")
    parser.add_argument("--fig-width", type=float, default=None,
        help="Total figure width in inches. ECCV textwidth=6.7. None=auto (2.2in/cell).")
    parser.add_argument("--row-label-x", type=float, default=0.5, metavar="X",
        help="Horizontal position of row label: 0=figure edge, 1=first subplot. Default 0.5 (midpoint).")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--out", default="output/panel.pdf", help="Output file (.pdf or .png)")

    # --- Model / checkpoint args (from-ckpts mode) --------------------------
    parser.add_argument(
        "--ckpts", nargs=3, type=parse_ckpt_arg, metavar="CKPT",
        help='3 checkpoints "Label:path" for the 3 methods',
    )
    parser.add_argument("--model", default="SiT-XL/2")
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--timestep", type=float, default=0.3)
    parser.add_argument("--class-label", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument(
        "--viz-mode", default="attn_output",
        choices=["attn_weights", "attn_output", "feature_sim"],
    )

    args = parser.parse_args()

    # Validate
    if args.ckpts is None:
        parser.error("--ckpts with 3 checkpoint paths is required.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load VAE ----------------------------------------------------------
    print("Loading VAE …")
    vae, latents_scale, latents_bias = load_vae(device)

    # ---- Load models -------------------------------------------------------
    need_proj = (args.viz_mode == "feature_sim")
    models_info = []
    for label, ckpt_path in args.ckpts:
        print(f"Loading [{label}] from {ckpt_path} …")
        model, enc_depth = load_sit_model(
            ckpt_path, args.model, device, args.resolution,
            need_projector=need_proj,
        )
        models_info.append((label, model, enc_depth))

    layer = (models_info[0][2] - 1) if need_proj else args.layer
    print(f"Using layer {layer}")

    # ---- Process each of the 4 cells ---------------------------------------
    cell_data = []
    for cell_idx, (img_path, query_str) in enumerate(zip(args.images, args.queries)):
        print(f"\n--- Cell {cell_idx}: {img_path}, query={query_str} ---")
        pixel_tensor, orig_img = load_image(img_path, size=args.resolution)
        latent = encode_to_latent(pixel_tensor, vae, latents_scale, latents_bias, device)

        latent_h = latent.shape[-2]
        patch_size = 2
        grid_size = latent_h // patch_size

        q_idx = resolve_query(query_str, grid_size)
        img_w, img_h = orig_img.size
        qy_pix = (q_idx // grid_size + 0.5) * (img_h / grid_size)
        qx_pix = (q_idx %  grid_size + 0.5) * (img_w / grid_size)
        qy_grid = q_idx // grid_size + 0.5
        qx_grid = q_idx %  grid_size + 0.5

        heatmaps = []
        for (mlabel, model, enc_depth) in models_info:
            cur_layer = (enc_depth - 1) if need_proj else layer
            print(f"  Extracting [{mlabel}] layer {cur_layer} …")
            data = extract_from_layer(
                model, latent, args.timestep, args.class_label, cur_layer,
                viz_mode=args.viz_mode,
            )
            if need_proj and hasattr(model, "projectors") and len(model.projectors) > 0:
                with torch.no_grad():
                    data = model.projectors[0](data.to(device)).detach().cpu()
            hmap = compute_heatmap(data, q_idx, grid_size, args.viz_mode, label=mlabel)
            heatmaps.append(hmap)

        cell_data.append({
            "orig": orig_img,
            "heatmaps": heatmaps,
            "q_pix": (qx_pix, qy_pix),
            "q_grid": (qx_grid, qy_grid),
        })

    # ---- Assemble panel ----------------------------------------------------
    assemble_panel(
        cell_data,
        method_labels=args.labels,
        row_label=args.row_label,
        group_labels=args.group_labels,
        save_path=args.out,
        cmap=args.cmap,
        row_label_x=args.row_label_x,
        font_family=args.font,
        fontsize=args.fontsize,
        fig_width=args.fig_width,
    )


if __name__ == "__main__":
    main()
