"""
Attention Map Visualization for Paper Figures.

Generates a publication-quality grid comparing attention heatmaps across
different training methods (e.g., Vanilla / REPA / Scaffolding) and query
positions.  All checkpoints are loaded as vanilla SiT (since at inference
time every method uses its own learned K/V).

Usage example (2 queries × 3 methods):
    CUDA_VISIBLE_DEVICES=5 python visualize_attention.py \
    --image images/dog.jpg \
    --ckpts "/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
            "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
    --model SiT-XL/2 \
    --query-region "4,4:8,8" \
    --viz-mode attn_output \
    --layer 4 \
    --timestep 0.1 \
    --out dog_attn_0.1_1.pdf

"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Imports from the SiT codebase
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sit import SiT_models
from models.autoencoder import VAE_F8D4
from torchvision import transforms


# ========================== Image helpers ==================================

def center_crop_arr(pil_image, image_size):
    """Center-crop and resize to a square, following the SiT training recipe."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(
        arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def load_image(image_path, size=256):
    """Load an image, center-crop, and return both the tensor and PIL image."""
    img = Image.open(image_path).convert("RGB")
    img = center_crop_arr(img, size)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    x = transform(img).unsqueeze(0)  # (1, 3, H, W)
    return x, img


# ========================== VAE helpers ====================================

def encode_to_latent(pixel_tensor, vae, latents_scale, latents_bias, device):
    """
    Encode a pixel-space image tensor (1, 3, H, W) into the normalised
    latent space (1, 4, H/8, W/8) expected by SiT.
    """
    with torch.no_grad():
        pixel_tensor = pixel_tensor.to(device)
        posterior = vae.encode(pixel_tensor)
        # Sample from posterior (mean + std * noise)
        z = posterior.sample()
        # Apply the same normalization as training
        z = (z - latents_bias) * latents_scale
    return z


# ========================== Model loading ==================================

def load_vae(device):
    """Load VAE and latent-space normalization statistics."""
    vae = VAE_F8D4().to(device).eval()
    vae_ckpt = torch.load(
        "pretrained_models/sdvae-ft-mse-f8d4.pt",
        map_location=device, weights_only=False,
    )
    vae.load_state_dict(vae_ckpt)

    latents_stats = torch.load(
        "pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt",
        map_location=device, weights_only=False,
    )
    latents_scale = latents_stats["latents_scale"].to(device).view(1, -1, 1, 1)
    latents_bias  = latents_stats["latents_bias"].to(device).view(1, -1, 1, 1)
    return vae, latents_scale, latents_bias


def load_sit_model(ckpt_path, model_name, device, resolution=256, need_projector=False):
    """
    Load a SiT model from a checkpoint.  When need_projector=True, the
    projector head is kept so we can map block outputs into the alignment
    space for feature-similarity visualisation.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Auto-detect settings from checkpoint args
    ckpt_args = ckpt.get("args", None)
    qk_norm = False
    encoder_depth = 8
    projection_layer_type = "conv"
    proj_kwargs_kernel_size = 1
    z_dims = [768]

    if ckpt_args is not None:
        if hasattr(ckpt_args, "qk_norm"):
            qk_norm = ckpt_args.qk_norm
        if hasattr(ckpt_args, "encoder_depth"):
            encoder_depth = ckpt_args.encoder_depth
        if hasattr(ckpt_args, "projection_layer_type"):
            projection_layer_type = ckpt_args.projection_layer_type
        if hasattr(ckpt_args, "proj_kwargs_kernel_size"):
            proj_kwargs_kernel_size = ckpt_args.proj_kwargs_kernel_size

    latent_size = resolution // 8
    block_kwargs = {"fused_attn": False, "qk_norm": qk_norm}

    # eval_mode=False when we need projector, so it gets created
    model = SiT_models[model_name](
        input_size=latent_size,
        in_channels=4,
        num_classes=1000,
        use_cfg=False,
        eval_mode=(not need_projector),
        z_dims=z_dims,
        encoder_depth=encoder_depth,
        projection_layer_type=projection_layer_type,
        proj_kwargs_kernel_size=proj_kwargs_kernel_size,
        **block_kwargs,
    )

    # Pick EMA weights if available, else raw model weights
    if "ema" in ckpt:
        state_dict = ckpt["ema"]
    elif "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    # Filter out encoder-KV keys that don't belong to vanilla SiT
    filtered = {k: v for k, v in state_dict.items() if "kv_proj" not in k}
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    if missing:
        print(f"  [info] {len(missing)} missing keys in {os.path.basename(ckpt_path)}: {missing[:5]}...")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys in {os.path.basename(ckpt_path)}")

    model.eval().to(device)
    return model, encoder_depth


# ========================== Attention / feature extraction =================

def extract_from_layer(model, latent, timestep, class_label, layer_idx,
                       viz_mode="feature_sim"):
    """
    Run a forward pass and intercept the target transformer block.

    Args:
        viz_mode:
          - "attn_weights":  capture softmax(Q @ K^T) → (1, heads, N, N)
          - "attn_output":   capture attn @ V (pre-proj) → (1, N, C)
          - "feature_sim":   capture block output → (1, N, C)
                             (to be projected externally)

    Returns:
        If attn_weights:  Tensor (1, heads, N, N)
        If attn_output:   Tensor (1, N, C)  — attention-aggregated features
        If feature_sim:   Tensor (1, N, C)  — full block hidden states
    """
    device = latent.device
    t = torch.tensor([timestep], device=device, dtype=torch.float32)
    y = torch.tensor([class_label], device=device, dtype=torch.long)

    target_block = model.blocks[layer_idx]
    captured = []

    if viz_mode in ("attn_weights", "attn_output"):
        # Hook into the attention module
        attn_module = target_block.attn
        original_forward = attn_module.forward

        def _patched_attn_forward(x_in, attn_mask=None):
            B, N, C = x_in.shape
            qkv = attn_module.qkv(x_in).reshape(
                B, N, 3, attn_module.num_heads, attn_module.head_dim
            ).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_module.q_norm(q), attn_module.k_norm(k)

            q = q * attn_module.scale
            attn = q @ k.transpose(-2, -1)          # (B, heads, N, N)
            attn = attn.softmax(dim=-1)

            x_out = attn @ v                         # (B, heads, N, head_dim)
            x_out = x_out.transpose(1, 2).reshape(B, N, C)

            if viz_mode == "attn_weights":
                captured.append(attn.detach().cpu())
            else:  # attn_output
                captured.append(x_out.detach().cpu())

            if hasattr(attn_module, 'norm'):
                x_out = attn_module.norm(x_out)
            x_out = attn_module.proj(x_out)
            if hasattr(attn_module, 'proj_drop'):
                x_out = attn_module.proj_drop(x_out)
            return x_out

        attn_module.forward = _patched_attn_forward
        restore = lambda: setattr(attn_module, 'forward', original_forward)
    else:
        # feature_sim: hook the block to capture full hidden states
        original_forward = target_block.forward

        def _patched_block_forward(x_in, c):
            x_out = original_forward(x_in, c)
            captured.append(x_out.detach().cpu())
            return x_out

        target_block.forward = _patched_block_forward
        restore = lambda: setattr(target_block, 'forward', original_forward)

    try:
        with torch.no_grad():
            model(latent, t, y)
    finally:
        restore()

    if not captured:
        raise RuntimeError("Failed to capture – check layer_idx.")
    return captured[0]


# ========================== Query position helpers =========================

def resolve_query(query_str, grid_size):
    """
    Convert a query specifier to an integer token index.

    Accepted values:
      - "center"          → token at the grid center
      - "top-left"        → token near top-left corner
      - "bottom-right"    → token near bottom-right corner
      - "row,col"         → grid coordinate, e.g. "8,8" or "4,12"
      - an integer string → flat token index (clamped to valid range)
    """
    total = grid_size * grid_size
    if query_str == "center":
        return (grid_size // 2) * grid_size + (grid_size // 2)
    elif query_str == "top-left":
        return (grid_size // 4) * grid_size + (grid_size // 4)
    elif query_str == "bottom-right":
        return (3 * grid_size // 4) * grid_size + (3 * grid_size // 4)
    elif "," in query_str:
        # Grid coordinate format: "row,col"
        parts = query_str.split(",")
        r, c = int(parts[0]), int(parts[1])
        r = min(max(r, 0), grid_size - 1)
        c = min(max(c, 0), grid_size - 1)
        return r * grid_size + c
    else:
        idx = int(query_str)
        return min(max(idx, 0), total - 1)


# ========================== Plotting =======================================

def plot_grid(
    original_img,
    attn_dict,           # {method_label: [data_for_q0, data_for_q1, ...]}
    query_indices,
    query_labels,
    grid_size,
    save_path,
    viz_mode="feature_sim",
    cmap="viridis",
):
    """
    Draw an N_queries × (1 + N_methods) grid in paper style.

    Column 0: original image with red ★ query marker.
    Columns 1..N_methods: pure attention heatmaps (no image overlay)
                          with red ★ showing the query position.
    A shared colorbar is placed on the right.
    """
    method_labels = list(attn_dict.keys())
    n_queries = len(query_indices)
    n_methods = len(method_labels)
    n_cols = 1 + n_methods  # first column is the original image

    # --- Pre-compute all heatmaps to find global vmin/vmax ----------------
    all_heatmaps = {}  # (label, row) → 2D numpy array
    for row, q_idx in enumerate(query_indices):
        for label in method_labels:
            data = attn_dict[label][row]
            if viz_mode == "attn_weights":
                attn_map = data[0].mean(dim=0).numpy()  # (N, N)
                query_attn = attn_map[q_idx]
            else:
                features = data[0]
                query_feat = features[q_idx]
                query_attn = F.cosine_similarity(
                    query_feat.unsqueeze(0), features, dim=-1
                ).numpy()
            all_heatmaps[(label, row)] = query_attn.reshape(grid_size, grid_size)

    # No normalization — use raw values; matplotlib auto-scales per imshow
    # For cosine similarity this means raw [-1, 1] range

    # --- Plot -------------------------------------------------------------
    fig = plt.figure(figsize=(3.0 * n_cols + 0.6, 3.0 * n_queries))

    gs = gridspec.GridSpec(
        n_queries, n_cols + 1,
        width_ratios=[1] * n_cols + [0.05],
        wspace=0.08, hspace=0.12,
    )

    img_w, img_h = original_img.size  # PIL: (W, H)
    all_ims = []

    for row, (q_idx, q_label) in enumerate(zip(query_indices, query_labels)):
        qy_pix = (q_idx // grid_size + 0.5) * (img_h / grid_size)
        qx_pix = (q_idx %  grid_size + 0.5) * (img_w / grid_size)
        qy_grid = q_idx // grid_size + 0.5
        qx_grid = q_idx %  grid_size + 0.5

        # --- Column 0: original image + query marker -----------------------
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(original_img)
        ax.plot(
            qx_pix, qy_pix,
            marker="*", color="red", markersize=14,
            markeredgecolor="darkred", markeredgewidth=0.8,
        )
        ax.set_ylabel(q_label, fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title("Input", fontsize=12, fontweight="bold")

        # --- Columns 1+: heatmaps with shared global color range ----------
        for col, label in enumerate(method_labels, start=1):
            ax = fig.add_subplot(gs[row, col])
            heatmap_2d = all_heatmaps[(label, row)]

            im = ax.imshow(
                heatmap_2d, cmap=cmap,
                interpolation="nearest", aspect="equal",
            )
            all_ims.append(im)

            ax.plot(
                qx_grid, qy_grid,
                marker="*", color="red", markersize=10,
                markeredgecolor="darkred", markeredgewidth=0.6,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(label, fontsize=12, fontweight="bold")

    # Shared colorbar on the right
    cbar_ax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(all_ims[0], cax=cbar_ax)
    cbar_label = "Attention Weight" if viz_mode == "attn_weights" else "Cosine Similarity"
    cbar.set_label(cbar_label, fontsize=10)

    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()
    print(f"Saved attention grid → {save_path}")


# ========================== CLI ============================================

def parse_ckpt_arg(s):
    """Parse 'Label:path' or plain 'path' into (label, path).
    If no label is given, infer from the parent experiment directory name.
    """
    if ":" in s and not s.startswith("/"):
        label, path = s.split(":", 1)
        return label.strip(), path.strip()
    # No label — auto-extract from path, e.g. ".../exps/vanilla_sit/checkpoints/0100000.pt" → "vanilla_sit"
    path = s.strip()
    parts = path.replace("\\", "/").split("/")
    # Try to find the directory right before "checkpoints"
    for i, p in enumerate(parts):
        if p == "checkpoints" and i > 0:
            return parts[i - 1], path
    # Fallback: use filename without extension
    return os.path.splitext(os.path.basename(path))[0], path


def main():
    parser = argparse.ArgumentParser(
        description="Attention map grid visualization for paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", type=str, nargs="+", required=True,
                        help="Path(s) to input RGB image(s)")
    parser.add_argument("--ckpts", type=parse_ckpt_arg, nargs="+", required=True,
                        help='One or more "Label:checkpoint.pt" pairs')
    parser.add_argument("--model", type=str, default="SiT-XL/2",
                        choices=list(SiT_models.keys()),
                        help="SiT architecture name")
    parser.add_argument("--layer", type=int, default=14,
                        help="Transformer block index for attn_weights mode (0-based). "
                             "Ignored in feature_sim mode (auto-uses encoder_depth).")
    parser.add_argument("--viz-mode", type=str, default="feature_sim",
                        choices=["attn_weights", "attn_output", "feature_sim"],
                        help="attn_weights: raw Q@K attention; "
                             "attn_output: cosine sim of attn@V (pre-proj); "
                             "feature_sim: cosine sim of projected block output")
    parser.add_argument("--queries", type=str, nargs="+",
                        default=["center", "top-left"],
                        help='Query positions: "center", "top-left", '
                             '"bottom-right", or integer token index')
    parser.add_argument("--query-region", type=str, default=None,
                        help='Region of queries: "r1,c1:r2,c2" expands to all '
                             'grid positions in [r1..r2, c1..c2]. '
                             'Overrides --queries. E.g. "4,4:8,8"')
    parser.add_argument("--timestep", type=float, nargs="+", default=[0.5],
                        help="Diffusion timestep(s) t ∈ [0,1], 0=clean, 1=noise. "
                             "Multiple values generate separate output files.")
    parser.add_argument("--class-label", type=int, default=0,
                        help="ImageNet class label (default 0)")
    parser.add_argument("--resolution", type=int, default=256,
                        choices=[256, 512],
                        help="Image resolution")
    parser.add_argument("--cmap", type=str, default="viridis",
                        help="Matplotlib colormap")
    parser.add_argument("--out", type=str, default="output/",
                        help="Output directory (e.g. 'output/' → "
                             "'output/{image}_{viz-mode}_t{timestep}_L{layer}.pdf')")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load VAE ----------------------------------------------------------
    print("Loading VAE...")
    vae, latents_scale, latents_bias = load_vae(device)

    # --- Load each checkpoint ------------------------------------------------
    need_proj = (args.viz_mode == "feature_sim")
    models_info = []
    for label, ckpt_path in args.ckpts:
        print(f"\nLoading [{label}] from {ckpt_path} ...")
        model, enc_depth = load_sit_model(
            ckpt_path, args.model, device, args.resolution,
            need_projector=need_proj,
        )
        layer = (enc_depth - 1) if need_proj else args.layer
        print(f"  Will extract from layer {layer} (encoder_depth={enc_depth})")
        models_info.append((label, model, enc_depth, layer))

    # --- Output directory ----------------------------------------------------
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    # --- Loop over images × timesteps --------------------------------------
    for image_path in args.image:
        img_name = os.path.splitext(os.path.basename(image_path))[0]
        print(f"\n{'#'*60}")
        print(f"Image: {image_path} ({img_name})")
        print(f"{'#'*60}")

        pixel_tensor, original_img = load_image(image_path, size=args.resolution)
        latent = encode_to_latent(pixel_tensor, vae, latents_scale, latents_bias, device)
        print(f"Latent shape: {latent.shape}")

        # --- Resolve query positions --------------------------------------
        latent_h = latent.shape[-2]
        patch_size = 2  # SiT default
        grid_size = latent_h // patch_size
        total_tokens = grid_size * grid_size

        # Expand --query-region "r1,c1:r2,c2" into individual queries
        queries = list(args.queries)  # copy so we don't mutate across images
        if args.query_region is not None:
            tl, br = args.query_region.split(":")
            r1, c1 = int(tl.split(",")[0]), int(tl.split(",")[1])
            r2, c2 = int(br.split(",")[0]), int(br.split(",")[1])
            r1, r2 = min(r1, r2), max(r1, r2)
            c1, c2 = min(c1, c2), max(c1, c2)
            queries = [f"{r},{c}" for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]
            print(f"Query region ({r1},{c1}):({r2},{c2}) → {len(queries)} queries")

        query_indices = [resolve_query(q, grid_size) for q in queries]
        query_labels = []
        for q_str, q_idx in zip(queries, query_indices):
            r, c = q_idx // grid_size, q_idx % grid_size
            query_labels.append(f"Query ({r},{c})")

        print(f"Grid: {grid_size}×{grid_size} = {total_tokens} tokens")
        print(f"Queries: {list(zip(queries, query_indices))}")

        # --- Loop over timesteps -------------------------------------------
        for timestep in args.timestep:
            print(f"\n{'='*60}")
            print(f"Timestep = {timestep}")
            print(f"{'='*60}")

            attn_dict = {}
            for label, model, enc_depth, layer in models_info:
                print(f"  Extracting [{label}] layer {layer} ...")
                data = extract_from_layer(
                    model, latent, timestep, args.class_label, layer,
                    viz_mode=args.viz_mode,
                )
                if need_proj and hasattr(model, 'projectors') and len(model.projectors) > 0:
                    with torch.no_grad():
                        data = model.projectors[0](data.to(device)).detach().cpu()
                    print(f"  Projected to alignment space: {data.shape}")
                attn_dict[label] = [data] * len(query_indices)

            layer_tag = models_info[0][3]
            save_path = os.path.join(out_dir, f"{img_name}_{args.viz_mode}_t{timestep}_L{layer_tag}.pdf")

            plot_grid(
                original_img, attn_dict,
                query_indices, query_labels,
                grid_size, save_path,
                viz_mode=args.viz_mode, cmap=args.cmap,
            )

    # Free memory
    del vae
    for _, model, _, _ in models_info:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
