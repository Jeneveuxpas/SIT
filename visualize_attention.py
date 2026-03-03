"""
Attention Map Side-by-Side Comparison
DINOv2 (teacher) | Vanilla DiT | iREPA | Ours

依赖：
    pip install torch torchvision timm matplotlib einops
    pip install git+https://github.com/facebookresearch/dinov2.git  # DINOv2
使用方法：
    CUDA_VISIBLE_DEVICES=7 python visualize_attention.py \
        --image "/workspace/SIT/images/dog.jpg" \
        --vae_ckpt "/workspace/SIT/pretrained_models/sdvae-ft-mse-f8d4.pt" \
        --dit_vanilla  "/workspace/SIT/exps/vanilla_sit/checkpoints/0100000.pt" \
        --dit_irepa    "/workspace/iREPA/ldm/exps/irepa_conv_1.0/checkpoints/0100000.pt" \
        --dit_ours     "/workspace/SIT/exps/conv_3_kv_2.0/checkpoints/0100000.pt" \
        --output       attention_comparison.pdf
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

def load_image(path: str, img_size: int = 224) -> tuple[torch.Tensor, np.ndarray]:
    """
    返回:
        tensor : (1, 3, H, W)  归一化后的输入
        vis    : (H, W, 3)     uint8 原图，用于背景展示
    """
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
    tensor = transform(img).unsqueeze(0)          # (1,3,H,W)
    vis    = (vis_transform(img).permute(1,2,0).numpy() * 255).astype(np.uint8)
    return tensor, vis


# ─────────────────────────────────────────────
# 2. 提取 DINOv2 的 attention
# ─────────────────────────────────────────────

def load_dinov2(model_name: str = "dinov2_vitb14") -> torch.nn.Module:
    """从 torch.hub 加载 DINOv2，无需本地安装"""
    model = torch.hub.load("facebookresearch/dinov2", model_name)
    model.eval()
    return model


def get_dinov2_attention(
    model: torch.nn.Module,
    x: torch.Tensor,            # (1, 3, H, W)
    patch_size: int = 14,
) -> np.ndarray:
    """
    提取 DINOv2 最后一个 block 中所有 heads 的 CLS→patch attention，
    返回 entropy 最低（最聚焦）那个 head 的 2D attention map。

    DINOv2 官方 Attention 的 attn_drop 是 float 而非 nn.Dropout，
    且使用 F.scaled_dot_product_attention 不暴露 attention weights，
    因此 hook qkv 线性层的输出，手动计算 attention。

    返回: (h_feat, w_feat) float32，已归一化到 [0,1]
    """
    qkv_store = []

    def hook(module, inp, out):
        qkv_store.append(out.detach())

    # hook qkv 投影层，捕获 (B, N, 3*D) 输出
    handle = model.blocks[-1].attn.qkv.register_forward_hook(hook)

    with torch.no_grad():
        model(x)
    handle.remove()

    # 手动计算 attention weights
    last_attn = model.blocks[-1].attn
    num_heads = last_attn.num_heads
    qkv = qkv_store[0]                                  # (1, N+1, 3*D)
    B, N, _ = qkv.shape
    head_dim = qkv.shape[-1] // (3 * num_heads)
    qkv = qkv.reshape(B, N, 3, num_heads, head_dim)     # (1, N+1, 3, H, d)
    q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)     # 各 (1, H, N+1, d)

    scale = head_dim ** -0.5
    attn = (q @ k.transpose(-2, -1)) * scale             # (1, H, N+1, N+1)
    attn = attn.softmax(dim=-1)
    attn = attn[0]                                        # (H, N+1, N+1)

    # CLS token attend to patch tokens
    cls_attn = attn[:, 0, 1:]     # (num_heads, N_patches)

    # 选 entropy 最低的 head（最聚焦）
    probs   = cls_attn / cls_attn.sum(-1, keepdim=True).clamp(min=1e-8)
    entropy = -(probs * (probs + 1e-8).log()).sum(-1)   # (num_heads,)
    best    = entropy.argmin().item()

    h_feat = x.shape[-2] // patch_size
    w_feat = x.shape[-1] // patch_size
    best_attn = cls_attn[best].reshape(h_feat, w_feat).cpu().numpy()

    # 归一化
    best_attn = (best_attn - best_attn.min()) / (best_attn.max() - best_attn.min() + 1e-8)
    return best_attn, best


# ─────────────────────────────────────────────
# 3. 提取 DiT / SiT 的 attention
# ─────────────────────────────────────────────

def get_dit_attention(
    model: torch.nn.Module,
    z: torch.Tensor,             # (1, 4, 32, 32) VAE latent
    timestep: float = 0.05,      # 低噪声，spatial 结构最清晰
    class_label: int = 0,
    block_idx: int = 8,          # 中间偏前的 block
    anchor: int = -1,            # anchor patch 索引，-1 表示中心点
    device: str = "cuda",
) -> np.ndarray:
    """
    对 DiT/SiT 模型提取指定 block 的 self-attention。
    支持 vanilla DiT、iREPA、Ours（inference 时结构相同）。

    输入必须是 VAE latent (1, 4, 32, 32)。

    返回: (h_feat, w_feat) float32，已归一化到 [0,1]
    """
    # 必须关闭 fused_attn，否则 attn_drop hook 捕获不到 attention weights
    old_fused = []
    for blk in model.blocks:
        old_fused.append(blk.attn.fused_attn)
        blk.attn.fused_attn = False

    attn_store = []

    def hook(module, inp, out):
        attn_store.append(out.detach())

    handle = model.blocks[block_idx].attn.attn_drop.register_forward_hook(hook)

    t_tensor = torch.tensor([timestep], device=device)
    y_tensor = torch.tensor([class_label], device=device)

    # 加轻微噪声模拟去噪中间步
    z = z.to(device)
    noise = torch.randn_like(z) * timestep
    z_noisy = z + noise

    with torch.no_grad():
        model(z_noisy, t_tensor, y_tensor)
    handle.remove()

    # 恢复 fused_attn 状态
    for blk, old in zip(model.blocks, old_fused):
        blk.attn.fused_attn = old

    attn = attn_store[0]           # (1, num_heads, N, N)
    attn = attn[0]                 # (num_heads, N, N)

    # DiT 没有 CLS token，N = (latent_size / patch_size)^2
    N = attn.shape[-1]
    if anchor < 0 or anchor >= N:
        anchor = N // 2

    anchor_attn = attn[:, anchor, :]   # (num_heads, N)

    # anchor_attn 已经过 softmax，直接算 entropy
    entropy = -(anchor_attn * (anchor_attn + 1e-8).log()).sum(-1)
    best    = entropy.argmin().item()

    h_feat  = w_feat = int(N ** 0.5)
    best_attn = anchor_attn[best].reshape(h_feat, w_feat).cpu().numpy()
    best_attn = (best_attn - best_attn.min()) / (best_attn.max() - best_attn.min() + 1e-8)
    return best_attn


# ─────────────────────────────────────────────
# 4. 绘图：4列并排
# ─────────────────────────────────────────────

COLUMN_TITLES = [
    "DINOv2\n(Teacher)",
    "Vanilla DiT",
    "iREPA",
    "Ours",
]
CMAP = "inferno"   # 论文常用：inferno / magma / jet


def plot_attention_comparison(
    original_img   : np.ndarray,      # (H, W, 3) uint8
    attn_maps      : list,            # list of 4 × (h_feat, w_feat) float32
    save_path      : str = "attention_comparison.pdf",
    img_size       : int = 224,
    dpi            : int = 300,
):
    """
    生成 2行 × 4列 的对比图：
      第 1 行：原图（4列相同，作为参考）
      第 2 行：各方法的 attention heatmap 叠加在原图上
    """
    assert len(attn_maps) == 4, "需要恰好 4 个 attention map"

    fig = plt.figure(figsize=(4 * 3.2, 2 * 3.0))
    gs  = gridspec.GridSpec(
        2, 4,
        figure=fig,
        wspace=0.04,
        hspace=0.06,
        left=0.01, right=0.99,
        top=0.88,  bottom=0.02,
    )

    for col, (title, attn) in enumerate(zip(COLUMN_TITLES, attn_maps)):

        # ── 第 1 行：原图 ──────────────────────────────
        ax_img = fig.add_subplot(gs[0, col])
        ax_img.imshow(original_img)
        ax_img.set_title(title, fontsize=11, fontweight="bold", pad=4)
        ax_img.axis("off")

        # ── 第 2 行：attention overlay ─────────────────
        ax_attn = fig.add_subplot(gs[1, col])

        # 上采样 attention map 到原图尺寸
        attn_tensor = torch.tensor(attn).unsqueeze(0).unsqueeze(0)   # (1,1,h,w)
        attn_up = F.interpolate(
            attn_tensor, size=(img_size, img_size), mode="bilinear", align_corners=False
        ).squeeze().numpy()

        # 原图作为背景（灰度化，减少颜色干扰）
        gray = np.mean(original_img, axis=-1, keepdims=True).repeat(3, axis=-1).astype(np.uint8)
        ax_attn.imshow(gray, alpha=0.45)
        im = ax_attn.imshow(attn_up, cmap=CMAP, alpha=0.75,
                            vmin=0.0, vmax=1.0, interpolation="bilinear")
        ax_attn.axis("off")

    # ── colorbar（整图共享）─────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.06, 0.015, 0.38])
    fig.colorbar(im, cax=cbar_ax)
    cbar_ax.tick_params(labelsize=8)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    print(f"[✓] 图像已保存至: {save_path}")
    plt.close()


# ─────────────────────────────────────────────
# 5. 主流程
# ─────────────────────────────────────────────

def load_vae(vae_ckpt: str, device: str) -> AutoencoderKL:
    """加载预训练 VAE (sd-vae-ft-ema 等)"""
    vae = AutoencoderKL(embed_dim=4, ch_mult=[1, 2, 4, 4], use_variational=True).to(device)
    state = torch.load(vae_ckpt, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def load_dit_checkpoint(ckpt_path: str, device: str) -> torch.nn.Module:
    """
    加载 SiT_XL_2 checkpoint，eval_mode=True 跳过 projector 创建。
    """
    model = SiT_XL_2(eval_mode=True, qk_norm=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # 兼容 "model" / "ema" / 裸 state_dict 三种格式
    if "model" in state:
        state = state["model"]
    elif "ema" in state:
        state = state["ema"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",       required=True,  help="输入图像路径")
    parser.add_argument("--vae_ckpt",    required=True,  help="VAE checkpoint 路径")
    parser.add_argument("--dit_vanilla", required=True,  help="Vanilla DiT checkpoint")
    parser.add_argument("--dit_irepa",   required=True,  help="iREPA checkpoint")
    parser.add_argument("--dit_ours",    required=True,  help="Ours checkpoint")
    parser.add_argument("--output",      default="attention_comparison.pdf")
    parser.add_argument("--block_idx",   type=int, default=8,  help="DiT block index")
    parser.add_argument("--anchor",      type=int, default=-1, help="anchor patch 索引 (0~255)，-1 为中心点")
    parser.add_argument("--timestep",    type=float, default=0.05)
    parser.add_argument("--class_label", type=int, default=0)
    parser.add_argument("--img_size",    type=int, default=256, help="输入图像尺寸 (VAE 输入)")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device

    # ① 加载图像
    print("[1/6] 加载图像 ...")
    x, vis = load_image(args.image, img_size=args.img_size)

    # ② DINOv2 attention
    print("[2/6] 提取 DINOv2 attention ...")
    dino = load_dinov2("dinov2_vitb14").to(device)
    # DINOv2 patch_size=14, 需要把图像 resize 到 14 的整数倍
    dino_size = (args.img_size // 14) * 14
    x_dino = F.interpolate(x, size=(dino_size, dino_size)).to(device)
    attn_dino, best_head = get_dinov2_attention(dino, x_dino, patch_size=14)
    print(f"    → DINOv2 选择 head {best_head}")
    del dino  # 释放显存

    # ③ VAE encode：RGB → latent
    print("[3/6] VAE 编码 ...")
    vae = load_vae(args.vae_ckpt, device)
    x_vae = F.interpolate(x, size=(args.img_size, args.img_size)).to(device)
    with torch.no_grad():
        posterior = vae.encode(x_vae)
        z_latent = posterior.sample()  # (1, 4, 32, 32)
    print(f"    → latent shape: {z_latent.shape}")
    del vae

    # ④ 加载 DiT 系列模型
    print("[4/6] 加载 DiT 模型 ...")
    dit_vanilla = load_dit_checkpoint(args.dit_vanilla, device)
    dit_irepa   = load_dit_checkpoint(args.dit_irepa,   device)
    dit_ours    = load_dit_checkpoint(args.dit_ours,    device)

    # ⑤ 提取 DiT attention（输入为 VAE latent）
    print("[5/6] 提取 DiT attention maps ...")
    common_kwargs = dict(
        timestep=args.timestep,
        class_label=args.class_label,
        block_idx=args.block_idx,
        anchor=args.anchor,
        device=device,
    )
    attn_vanilla = get_dit_attention(dit_vanilla, z_latent, **common_kwargs)
    attn_irepa   = get_dit_attention(dit_irepa,   z_latent, **common_kwargs)
    attn_ours    = get_dit_attention(dit_ours,    z_latent, **common_kwargs)

    # ⑥ 绘图
    print("[6/6] 绘制对比图 ...")
    attn_maps = [attn_dino, attn_vanilla, attn_irepa, attn_ours]
    plot_attention_comparison(
        original_img=vis,
        attn_maps=attn_maps,
        save_path=args.output,
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()