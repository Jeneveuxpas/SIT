"""
Attention Map Side-by-Side Comparison  [v3 - Unified Query Point]
DINOv2 (Teacher) | Vanilla DiT | iREPA | Ours

核心改动（相比v2）：
  - DINOv2 和 DiT 统一使用同一个 query patch（而非CLS token）
  - 所有模型都是：attn[anchor_patch, :] 平均所有 head
  - anchor 通过 --anchor_x / --anchor_y 指定像素坐标（相对于256×256图像）
  - 可先不加坐标参数运行，脚本会在图上标出网格帮你选点

使用方法：
    # Step1: 先看网格，选一个语义强的点（如狗鼻子）
    CUDA_VISIBLE_DEVICES=7 python visualize_attention_v3.py \
        --image "/workspace/SIT/images/dog.jpg" \
        --vae_ckpt "/workspace/SIT/pretrained_models/sdvae-ft-mse-f8d4.pt" \
        --dit_vanilla  "/workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt" \
        --dit_irepa    "/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
        --dit_ours     "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
        --preview_grid                  

    # Step2: 确认坐标后正式运行
    CUDA_VISIBLE_DEVICES=7 python visualize_attention_v3.py \
        --image "/workspace/SIT/images/dog.jpg" \
        --vae_ckpt "/workspace/SIT/pretrained_models/sdvae-ft-mse-f8d4.pt" \
        --dit_vanilla  "/workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt" \
        --dit_irepa    "/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
        --dit_ours     "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
        --anchor_x 120 \
        --anchor_y 80 \
        --output attention_comparison.pdf
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from PIL import Image
import torchvision.transforms as T

from models.sit import SiT_XL_2
from models.autoencoder import AutoencoderKL


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
CMAP = "inferno"
COLUMN_TITLES = ["DINOv2\n(Teacher)", "Vanilla DiT", "iREPA", "Ours"]


# ─────────────────────────────────────────────
# 工具：像素坐标 → patch 索引
# ─────────────────────────────────────────────

def pixel_to_patch(px, py, img_size, patch_size):
    """
    将像素坐标 (px, py) 转换为 patch 索引。
    px: 水平方向像素坐标 (0 ~ img_size-1)
    py: 垂直方向像素坐标 (0 ~ img_size-1)
    返回: (patch_row, patch_col, linear_index)
    """
    h_feat = img_size // patch_size
    w_feat = img_size // patch_size
    col = min(int(px / img_size * w_feat), w_feat - 1)
    row = min(int(py / img_size * h_feat), h_feat - 1)
    return row, col, row * w_feat + col


# ─────────────────────────────────────────────
# 图像加载
# ─────────────────────────────────────────────

def load_image(path, img_size=256):
    img = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    vis_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])
    tensor = tf(img).unsqueeze(0)
    vis    = (vis_tf(img).permute(1,2,0).numpy() * 255).astype(np.uint8)
    return tensor, vis


# ─────────────────────────────────────────────
# 预览：在图上画出 patch 网格，帮助选 anchor
# ─────────────────────────────────────────────

def preview_grid(vis_img, img_size, patch_size_dino=14, patch_size_dit=8,
                 save_path="grid_preview.pdf"):
    """
    输出一张图，叠加 DINOv2 和 DiT 的 patch 网格，
    帮助直观选择 anchor 的像素坐标。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, ps, title in zip(
        axes,
        [patch_size_dino, patch_size_dit],
        [f"DINOv2 grid (patch={patch_size_dino}px)",
         f"DiT grid (patch={patch_size_dit}px)"]
    ):
        ax.imshow(vis_img)
        h_feat = img_size // ps
        w_feat = img_size // ps
        # 画竖线
        for c in range(w_feat + 1):
            ax.axvline(c * ps, color="cyan", linewidth=0.5, alpha=0.7)
        # 画横线
        for r in range(h_feat + 1):
            ax.axhline(r * ps, color="cyan", linewidth=0.5, alpha=0.7)
        # 标注 patch 索引（每隔几个标一次避免拥挤）
        step = max(1, h_feat // 8)
        for r in range(0, h_feat, step):
            for c in range(0, w_feat, step):
                ax.text(c * ps + ps/2, r * ps + ps/2,
                        f"{r*w_feat+c}",
                        color="yellow", fontsize=5,
                        ha="center", va="center")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    fig.suptitle("选择 anchor patch：在图上找到目标语义点（如鼻子/眼睛），\n"
                 "读出其像素坐标 (px, py) 后用 --anchor_x px --anchor_y py 传入",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"[✓] 网格预览已保存: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# DINOv2：指定 query patch，head 平均
# ─────────────────────────────────────────────

def get_dinov2_attention(model, x, anchor_idx, patch_size=14):
    """
    统一接口：取 anchor_idx 这个 patch 作为 query，
    看它关注哪里。所有 head 平均。

    注意：anchor_idx 是在 DINOv2 的 patch 空间里的索引
    （token 0 是 CLS，所以实际取 attn[:, anchor_idx+1, 1:]）
    """
    qkv_store = []

    def hook(m, inp, out):
        qkv_store.append(out.detach())

    handle = model.blocks[-1].attn.qkv.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    num_heads = model.blocks[-1].attn.num_heads
    qkv = qkv_store[0]                              # (1, N+1, 3*D)
    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
    q, k, _ = qkv.permute(2, 0, 3, 1, 4).unbind(0) # (1, H, N+1, d)

    attn = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
    attn = attn.softmax(dim=-1)[0]                   # (H, N+1, N+1)

    # query = anchor patch（+1 因为 token 0 是 CLS）
    # key   = 所有 patch tokens（去掉 CLS）
    query_row = attn[:, anchor_idx + 1, 1:]          # (H, N_patches)

    # head 平均
    mean_attn = query_row.mean(dim=0)                # (N_patches,)

    h_feat = x.shape[-2] // patch_size
    w_feat = x.shape[-1] // patch_size
    attn_map = mean_attn.reshape(h_feat, w_feat).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


# ─────────────────────────────────────────────
# DiT：指定 query patch，head 平均
# ─────────────────────────────────────────────

def get_dit_attention(model, z, anchor_idx, timestep=0.3,
                      class_label=0, block_idx=8, device="cuda"):
    """
    取 anchor_idx 这个 patch 作为 query，看它关注哪里。
    所有 head 平均。
    """
    old_fused = []
    for blk in model.blocks:
        old_fused.append(getattr(blk.attn, 'fused_attn', False))
        if hasattr(blk.attn, 'fused_attn'):
            blk.attn.fused_attn = False

    attn_store = []

    def hook(m, inp, out):
        attn_store.append(out.detach())

    handle = model.blocks[block_idx].attn.attn_drop.register_forward_hook(hook)

    t_tensor = torch.tensor([timestep], device=device)
    y_tensor = torch.tensor([class_label], device=device)
    z_noisy  = z.to(device) + torch.randn_like(z) * timestep

    with torch.no_grad():
        model(z_noisy, t_tensor, y_tensor)
    handle.remove()

    for blk, old in zip(model.blocks, old_fused):
        if hasattr(blk.attn, 'fused_attn'):
            blk.attn.fused_attn = old

    attn = attn_store[0][0]   # (H, N, N)
    N    = attn.shape[-1]
    anchor_idx = min(anchor_idx, N - 1)

    # query = anchor patch，key = 所有 patch，head 平均
    query_row = attn[:, anchor_idx, :]   # (H, N)
    mean_attn = query_row.mean(dim=0)    # (N,)

    h_feat = w_feat = int(N ** 0.5)
    attn_map = mean_attn.reshape(h_feat, w_feat).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


# ─────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────

def plot_attention_comparison(original_img, attn_maps, anchor_px, anchor_py,
                               save_path="attention_comparison_v3.pdf",
                               img_size=256, dpi=300):
    assert len(attn_maps) == 4

    fig = plt.figure(figsize=(4 * 3.0, 2 * 2.8))
    gs  = gridspec.GridSpec(
        2, 4, figure=fig,
        wspace=0.03, hspace=0.04,
        left=0.01, right=0.93,
        top=0.90,  bottom=0.02,
    )

    im_ref = None
    for col, (title, attn) in enumerate(zip(COLUMN_TITLES, attn_maps)):

        # 第1行：原图 + 标出 anchor 点
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(original_img)
        ax_img.scatter(
            [anchor_px], [anchor_py],
            s=60, c="red", marker="*",
            zorder=5, linewidths=0.5, edgecolors="white"
        )
        ax_img.set_title(title, fontsize=11, fontweight="bold", pad=5)
        ax_img.axis("off")

        # 第2行：attention heatmap
        ax_attn = fig.add_subplot(gs[1, col])
        attn_t  = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
        attn_up = F.interpolate(
            attn_t, size=(img_size, img_size),
            mode="bilinear", align_corners=False
        ).squeeze().numpy()

        gray = np.mean(original_img, axis=-1, keepdims=True)\
                 .repeat(3, axis=-1).astype(np.uint8)
        ax_attn.imshow(gray, alpha=0.40)
        im_ref = ax_attn.imshow(
            attn_up, cmap=CMAP, alpha=0.80,
            vmin=0.0, vmax=1.0, interpolation="bilinear"
        )
        # 第2行也标出 anchor 位置
        ax_attn.scatter(
            [anchor_px], [anchor_py],
            s=60, c="red", marker="*",
            zorder=5, linewidths=0.5, edgecolors="white"
        )
        ax_attn.axis("off")

    cbar_ax = fig.add_axes([0.945, 0.02, 0.018, 0.42])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[✓] 已保存: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 模型加载
# ─────────────────────────────────────────────

def load_vae(vae_ckpt, device):
    vae = AutoencoderKL(embed_dim=4, ch_mult=[1,2,4,4], use_variational=True).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def load_dit(ckpt_path, device):
    model = SiT_XL_2(eval_mode=True, qk_norm=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:   state = state["model"]
    elif "ema" in state:   state = state["ema"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",        required=True)
    parser.add_argument("--vae_ckpt",     default=None)
    parser.add_argument("--dit_vanilla",  default=None)
    parser.add_argument("--dit_irepa",    default=None)
    parser.add_argument("--dit_ours",     default=None)
    parser.add_argument("--output",       default="attention_comparison_v3.pdf")
    parser.add_argument("--anchor_x",     type=int, default=None,
                        help="anchor 的像素 x 坐标（水平方向）")
    parser.add_argument("--anchor_y",     type=int, default=None,
                        help="anchor 的像素 y 坐标（垂直方向）")
    parser.add_argument("--block_idx",    type=int,   default=8)
    parser.add_argument("--timestep",     type=float, default=0.3)
    parser.add_argument("--class_label",  type=int,   default=0)
    parser.add_argument("--img_size",     type=int,   default=256)
    parser.add_argument("--preview_grid", action="store_true",
                        help="只输出 patch 网格预览图，帮助选 anchor 坐标")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    x, vis = load_image(args.image, img_size=args.img_size)

    # ── 仅输出网格预览 ─────────────────────────────
    if args.preview_grid:
        preview_grid(vis, args.img_size,
                     patch_size_dino=14, patch_size_dit=8,
                     save_path="grid_preview.pdf")
        return

    # ── anchor 坐标处理 ────────────────────────────
    if args.anchor_x is None or args.anchor_y is None:
        # 默认：图像中心
        args.anchor_x = args.img_size // 2
        args.anchor_y = args.img_size // 2
        print(f"[!] 未指定 anchor，使用图像中心 ({args.anchor_x}, {args.anchor_y})")
        print(f"    建议先运行 --preview_grid 选择语义更强的点")

    # DINOv2 patch 空间的 anchor 索引
    dino_size   = (args.img_size // 14) * 14   # 向下对齐到14的倍数
    _, _, dino_anchor = pixel_to_patch(
        args.anchor_x * dino_size / args.img_size,   # 坐标缩放到 dino_size
        args.anchor_y * dino_size / args.img_size,
        dino_size, patch_size=14
    )
    # DiT latent patch 空间的 anchor 索引（latent 32×32，patch_size=2 → 16×16 patches）
    # 注意：DiT 的 spatial 对应原图 img_size，patch_size 取决于你的 SiT 配置
    dit_latent_size  = args.img_size // 8      # VAE 压缩比 8
    dit_patch_size   = 2                        # SiT-XL/2 的 /2 就是 patch_size=2
    dit_feat_size    = dit_latent_size // dit_patch_size   # 一般=16
    _, _, dit_anchor = pixel_to_patch(
        args.anchor_x * dit_feat_size * dit_patch_size / args.img_size,
        args.anchor_y * dit_feat_size * dit_patch_size / args.img_size,
        dit_feat_size * dit_patch_size, patch_size=dit_patch_size
    )

    print(f"[anchor] 像素({args.anchor_x},{args.anchor_y}) "
          f"→ DINOv2 patch {dino_anchor}, DiT patch {dit_anchor}")

    # ── DINOv2 ────────────────────────────────────
    print("[1] DINOv2 ...")
    dino   = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).eval()
    x_dino = F.interpolate(x, size=(dino_size, dino_size)).to(device)
    attn_dino = get_dinov2_attention(dino, x_dino, dino_anchor, patch_size=14)
    del dino

    # ── VAE encode ────────────────────────────────
    print("[2] VAE encode ...")
    vae      = load_vae(args.vae_ckpt, device)
    x_vae    = F.interpolate(x, size=(args.img_size, args.img_size)).to(device)
    with torch.no_grad():
        z_latent = vae.encode(x_vae).sample()
    del vae

    # ── DiT 系列 ──────────────────────────────────
    print("[3] DiT models ...")
    common = dict(anchor_idx=dit_anchor, timestep=args.timestep,
                  class_label=args.class_label, block_idx=args.block_idx,
                  device=device)
    attn_vanilla = get_dit_attention(load_dit(args.dit_vanilla, device), z_latent, **common)
    attn_irepa   = get_dit_attention(load_dit(args.dit_irepa,   device), z_latent, **common)
    attn_ours    = get_dit_attention(load_dit(args.dit_ours,    device), z_latent, **common)

    # ── 绘图 ──────────────────────────────────────
    print("[4] 绘图 ...")
    plot_attention_comparison(
        original_img = vis,
        attn_maps    = [attn_dino, attn_vanilla, attn_irepa, attn_ours],
        anchor_px    = args.anchor_x,
        anchor_py    = args.anchor_y,
        save_path    = args.output,
        img_size     = args.img_size,
    )


if __name__ == "__main__":
    main()