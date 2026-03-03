"""
Attention Map Side-by-Side Comparison  [v2 - Fixed]
DINOv2 (Teacher) | Vanilla DiT | iREPA | Ours

主要修复：
  1. DINOv2 改用 所有head平均，不再用 entropy-min（解决全黑问题）
  2. DiT timestep 从 0.05 → 0.3（attention 更平滑，更有语义）
  3. DiT 改用 所有head平均
  4. anchor 从固定中心点 → DINOv2 最高 attention 的 patch（自动找前景）
  5. colorbar 布局修复（统一放右侧，不遮挡图）

使用方法：
    CUDA_VISIBLE_DEVICES=7 python visualize_attention_v2.py \
        --image "/workspace/SIT/images/dog.jpg" \
        --vae_ckpt "/workspace/SIT/pretrained_models/sdvae-ft-mse-f8d4.pt" \
        --dit_vanilla  "/workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt" \
        --dit_irepa    "/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
        --dit_ours     "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
        --output       attention_comparison_v2.pdf
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
import torchvision.transforms as T

from models.sit import SiT_XL_2
from models.autoencoder import AutoencoderKL


# ─────────────────────────────────────────────
# 1. 图像预处理
# ─────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def load_image(path: str, img_size: int = 256):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    vis_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
    ])
    tensor = transform(img).unsqueeze(0)
    vis    = (vis_transform(img).permute(1,2,0).numpy() * 255).astype(np.uint8)
    return tensor, vis


# ─────────────────────────────────────────────
# 2. DINOv2 attention  【修复：改用全head平均】
# ─────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitb14", device: str = "cuda"):
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval().to(device)
    return model


def get_dinov2_attention(model, x, patch_size=14):
    """
    ✅ 修复：所有 head 平均，而非 entropy-min。
    entropy-min 会选出最尖锐（集中在1-2个patch）的head，视觉上全黑。
    平均后能展示完整的语义前景区域（=DINO原论文Figure 1风格）。

    返回:
        attn_map : (h_feat, w_feat) float32, 归一化到 [0,1]
        fg_anchor: int, 前景中心 patch 的索引（供 DiT 使用）
    """
    qkv_store = []

    def hook(module, inp, out):
        qkv_store.append(out.detach())

    handle = model.blocks[-1].attn.qkv.register_forward_hook(hook)
    with torch.no_grad():
        model(x)
    handle.remove()

    last_attn = model.blocks[-1].attn
    num_heads = last_attn.num_heads
    qkv = qkv_store[0]                              # (1, N+1, 3*D)
    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0) # (1, H, N+1, d)

    scale = head_dim ** -0.5
    attn  = (q @ k.transpose(-2, -1)) * scale       # (1, H, N+1, N+1)
    attn  = attn.softmax(dim=-1)[0]                  # (H, N+1, N+1)

    # CLS → patch attention: (H, N_patches)
    cls_attn = attn[:, 0, 1:]

    # ✅ 所有 head 平均（原来是 entropy-min，导致全黑）
    mean_attn = cls_attn.mean(dim=0)                 # (N_patches,)

    h_feat = x.shape[-2] // patch_size
    w_feat = x.shape[-1] // patch_size
    attn_map = mean_attn.reshape(h_feat, w_feat).cpu().numpy()

    # 找前景 anchor（attention 最高的 patch 索引，供 DiT 使用）
    fg_anchor = mean_attn.argmax().item()

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map, fg_anchor


# ─────────────────────────────────────────────
# 3. DiT attention  【修复：timestep↑ + head平均 + 语义anchor】
# ─────────────────────────────────────────────

def get_dit_attention(
    model,
    z,
    timestep  : float = 0.3,   # ✅ 修复：0.05→0.3，attention更平滑有语义
    class_label: int  = 0,
    block_idx  : int  = 8,
    anchor     : int  = -1,    # ✅ 修复：传入DINOv2前景anchor，不用固定中心
    device     : str  = "cuda",
):
    """
    ✅ 修复点：
      - timestep 0.05 → 0.3：低timestep下attention极度尖锐，
        随机指向纹理/噪声，而非语义。0.3 时空间结构更清晰。
      - 改用全 head 平均，比单个 head 更稳定。
      - anchor 从中心点改为 DINOv2 对应的前景 patch。
    """
    # 关闭 fused_attn 以便 hook 捕获 attention weights
    old_fused = []
    for blk in model.blocks:
        old_fused.append(getattr(blk.attn, 'fused_attn', False))
        if hasattr(blk.attn, 'fused_attn'):
            blk.attn.fused_attn = False

    attn_store = []
    def hook(module, inp, out):
        attn_store.append(out.detach())

    handle = model.blocks[block_idx].attn.attn_drop.register_forward_hook(hook)

    t_tensor = torch.tensor([timestep], device=device)
    y_tensor = torch.tensor([class_label], device=device)
    z = z.to(device)
    noise = torch.randn_like(z) * timestep
    z_noisy = z + noise

    with torch.no_grad():
        model(z_noisy, t_tensor, y_tensor)
    handle.remove()

    for blk, old in zip(model.blocks, old_fused):
        if hasattr(blk.attn, 'fused_attn'):
            blk.attn.fused_attn = old

    attn = attn_store[0][0]   # (num_heads, N, N)
    N = attn.shape[-1]

    # ✅ anchor：使用传入的 DINOv2 前景 anchor（映射到 DiT 的 token 空间）
    # DINOv2 在 dino_size×dino_size 上有 (dino_size/14)^2 个 patch
    # DiT latent 是 32×32/patch_size^2 个 patch
    # 这里直接用相对位置映射
    if anchor < 0 or anchor >= N:
        anchor_dit = N // 2
    else:
        # 将 DINOv2 的 2D 位置映射到 DiT 的 2D 位置
        h_dino = w_dino = int(anchor ** 0.5 + 0.5)  # 近似
        h_dit  = w_dit  = int(N ** 0.5)
        # 先还原 DINOv2 anchor 的 2D 坐标
        h_feat_dino = int(anchor ** 0 )   # 用下面更正确的方式
        # 直接按比例缩放
        anchor_2d_dino = np.unravel_index(anchor, (int(N**0.5), int(N**0.5)))
        anchor_dit = anchor_2d_dino[0] * h_dit + anchor_2d_dino[1]
        anchor_dit = min(anchor_dit, N - 1)

    # 取 anchor patch 对所有 patch 的 attention
    anchor_attn = attn[:, anchor_dit, :]   # (num_heads, N)

    # ✅ 所有 head 平均
    mean_attn = anchor_attn.mean(dim=0)    # (N,)

    h_feat = w_feat = int(N ** 0.5)
    attn_map = mean_attn.reshape(h_feat, w_feat).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


# ─────────────────────────────────────────────
# 4. 绘图  【修复：colorbar 布局】
# ─────────────────────────────────────────────

COLUMN_TITLES = ["DINOv2\n(Teacher)", "Vanilla DiT", "iREPA", "Ours"]
CMAP = "inferno"

def plot_attention_comparison(
    original_img,      # (H, W, 3) uint8
    attn_maps,         # list of 4 × (h_feat, w_feat)
    save_path = "attention_comparison_v2.pdf",
    img_size  = 256,
    dpi       = 300,
):
    assert len(attn_maps) == 4

    # ✅ 修复：缩小整图宽度，给 colorbar 留出合理空间
    fig = plt.figure(figsize=(4 * 3.0, 2 * 2.8))
    gs  = gridspec.GridSpec(
        2, 4,
        figure=fig,
        wspace=0.03,
        hspace=0.04,
        left=0.01, right=0.93,   # ✅ 右侧留0.07给colorbar
        top=0.90,  bottom=0.02,
    )

    im_ref = None
    for col, (title, attn) in enumerate(zip(COLUMN_TITLES, attn_maps)):

        # 第1行：原图
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(original_img)
        ax_img.set_title(title, fontsize=11, fontweight="bold", pad=5)
        ax_img.axis("off")

        # 第2行：attention overlay
        ax_attn = fig.add_subplot(gs[1, col])

        attn_tensor = torch.tensor(attn).unsqueeze(0).unsqueeze(0)
        attn_up = F.interpolate(
            attn_tensor, size=(img_size, img_size),
            mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # 背景：灰度原图
        gray = np.mean(original_img, axis=-1, keepdims=True).repeat(3, axis=-1).astype(np.uint8)
        ax_attn.imshow(gray, alpha=0.40)
        im_ref = ax_attn.imshow(
            attn_up, cmap=CMAP, alpha=0.80,
            vmin=0.0, vmax=1.0, interpolation="bilinear"
        )
        ax_attn.axis("off")

    # ✅ 修复：colorbar 放在整图右侧，位置与第2行对齐
    cbar_ax = fig.add_axes([0.945, 0.02, 0.018, 0.42])
    cbar = fig.colorbar(im_ref, cax=cbar_ax)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[✓] 已保存: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5. 模型加载
# ─────────────────────────────────────────────

def load_vae(vae_ckpt, device):
    vae = AutoencoderKL(embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def load_dit_checkpoint(ckpt_path, device):
    model = SiT_XL_2(eval_mode=True, qk_norm=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model" in state:
        state = state["model"]
    elif "ema" in state:
        state = state["ema"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# ─────────────────────────────────────────────
# 6. 主流程
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True)
    parser.add_argument("--vae_ckpt",    required=True)
    parser.add_argument("--dit_vanilla", required=True)
    parser.add_argument("--dit_irepa",   required=True)
    parser.add_argument("--dit_ours",    required=True)
    parser.add_argument("--output",      default="attention_comparison_v2.pdf")
    parser.add_argument("--block_idx",   type=int,   default=8)
    parser.add_argument("--timestep",    type=float, default=0.3)   # ✅ 改为0.3
    parser.add_argument("--class_label", type=int,   default=0)
    parser.add_argument("--img_size",    type=int,   default=256)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    device = args.device

    print("[1/6] 加载图像 ...")
    x, vis = load_image(args.image, img_size=args.img_size)

    print("[2/6] 提取 DINOv2 attention ...")
    dino   = load_dinov2("dinov2_vitb14", device)
    dino_size = (args.img_size // 14) * 14          # 向下取整到14的倍数
    x_dino = F.interpolate(x, size=(dino_size, dino_size)).to(device)
    attn_dino, fg_anchor = get_dinov2_attention(dino, x_dino, patch_size=14)
    print(f"    → DINOv2 前景 anchor patch: {fg_anchor}")
    del dino

    print("[3/6] VAE 编码 ...")
    vae  = load_vae(args.vae_ckpt, device)
    x_vae = F.interpolate(x, size=(args.img_size, args.img_size)).to(device)
    with torch.no_grad():
        z_latent = vae.encode(x_vae).sample()
    print(f"    → latent shape: {z_latent.shape}")
    del vae

    print("[4/6] 加载 DiT 模型 ...")
    dit_vanilla = load_dit_checkpoint(args.dit_vanilla, device)
    dit_irepa   = load_dit_checkpoint(args.dit_irepa,   device)
    dit_ours    = load_dit_checkpoint(args.dit_ours,    device)

    print("[5/6] 提取 DiT attention maps ...")
    common_kwargs = dict(
        timestep    = args.timestep,
        class_label = args.class_label,
        block_idx   = args.block_idx,
        anchor      = fg_anchor,           # ✅ 用DINOv2前景anchor
        device      = device,
    )
    attn_vanilla = get_dit_attention(dit_vanilla, z_latent, **common_kwargs)
    attn_irepa   = get_dit_attention(dit_irepa,   z_latent, **common_kwargs)
    attn_ours    = get_dit_attention(dit_ours,    z_latent, **common_kwargs)

    print("[6/6] 绘制对比图 ...")
    plot_attention_comparison(
        original_img = vis,
        attn_maps    = [attn_dino, attn_vanilla, attn_irepa, attn_ours],
        save_path    = args.output,
        img_size     = args.img_size,
    )


if __name__ == "__main__":
    main()