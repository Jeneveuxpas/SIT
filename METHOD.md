# SIT: Encoder K/V Distillation for Accelerated Diffusion Training

## Core Idea

在 SiT (flow-matching DiT) 的 attention 层中引入预训练视觉编码器（DINOv2）的 K/V，给模型提供**空间路由先验（spatial routing prior）**，缓解训练初期的 routing cold start，从而加速收敛。

## Two-Stage Training

### Stage 1: Encoder K/V Injection (Routing Bootstrap)

训练早期直接用 encoder 的 K/V 替代 SiT 自身 K/V：

```python
if stage == 1 and k_enc is not None:
    k = k_enc
    v = v_enc
```

这一步的目的不是长期替代 SiT attention，而是让 query 在 early stage 尽快学到可用的空间路由模式。

### Stage 2: Native K/V + Distillation

切回 SiT 自身 K/V，同时用 distillation loss 维持 Stage 1 到 Stage 2 的路由连续性：

```python
elif stage == 2 and k_enc is not None:
    distill_loss = compute_alignment(q, k_sit, v_sit, k_enc, v_enc)
    k = k_sit
    v = v_sit
```

```
Step:   0        stage1_steps                 total_steps
        |  Stage 1  |        Stage 2            |
KV:     | K_enc     |   K_sit + distill_loss    |
```

## Objective

总体目标：

$$
\mathcal{L}
=
\mathcal{L}_{\text{denoise}}
+
\lambda_{\text{repa}} \cdot g_{\text{repa}}(n)\cdot \mathcal{L}_{\text{REPA}}
+
\lambda_{\text{distill}} \cdot g_{\text{kv}}(n)\cdot \mathcal{L}_{\text{distill}}
$$

其中：
- $g_{\text{repa}}(n)$ 由 `repa-stop-step` / `repa-stop-fade-steps` 控制。
- $g_{\text{kv}}(n)$ 由 `kv-stop-step` / `kv-stop-fade-steps` 控制（仅 Stage 2 蒸馏分支）。

与代码一致：`repa_gate` 和 `kv_gate` 分别乘在 `proj_loss` 与 `distill_loss` 上。

## Loss Components

### 1. Denoising Loss

标准 v-prediction 目标（flow matching）。

### 2. REPA Projection Loss

对齐 SiT 中间层 hidden states 与 encoder patch features，提升中层表征质量。

### 3. Distillation Loss (Attention-Level)

主配置使用 `attn_mse`：

$$
\mathcal{L}_{\text{attn\_mse}}
=
\text{MSE}\left(\text{SDPA}(Q,K_{\text{sit}},V_{\text{sit}}),\ \text{SDPA}(Q,K_{\text{enc}},V_{\text{enc}})\right)
$$

```python
attn_enc = SDPA(Q, K_enc, V_enc)
attn_sit = SDPA(Q, K_sit, V_sit)
L_distill = MSE(attn_sit, attn_enc.detach())
```

## Encoder K/V Extraction and Projection

### Extraction via Forward Hooks

从 frozen encoder 指定层截取 K/V，无需改 encoder 主体：

```python
class EncoderKVExtractor:
    def forward(self, x_clean):
        _ = self.encoder(x_clean)      # trigger hooks
        return self.collected_kv_list  # [(K_l, V_l), ...]
```

### Projection for Dim/Head Alignment

DINOv2-B 与 SiT-XL 维度和 heads 不同，需要投影到 SiT attention space：

```python
class EncoderKVProjection(nn.Module):
    def forward(self, k_enc, v_enc, stage):
        k_proj = self.k_proj(k_enc)
        v_proj = self.v_proj(v_enc)
        if stage == 2:
            k_proj = k_proj.detach()
            v_proj = v_proj.detach()
        return k_proj, v_proj
```

关键设计：Stage 2 对投影输出 `detach`，distill 梯度只更新 SiT attention 分支，不再改投影映射。

## Training Loop (Implementation-Aligned)

```python
for step in training:
    stage = 1 if step < stage1_steps else 2

    repa_gate = stop_multiplier(step, repa_stop_step, repa_stop_fade_steps)
    kv_gate = stop_multiplier(step, kv_stop_step, kv_stop_fade_steps)

    repa_active = repa_loss and repa_gate > 0
    kv_active = use_kv and (stage == 1 or kv_gate > 0)

    with torch.no_grad():
        if repa_active or kv_active:
            enc_kv_list, zs = encoder_forward_once_if_possible(...)

    denoise_loss, proj_loss_raw, distill_loss_raw = forward_and_compute_losses(...)
    proj_loss = repa_gate * proj_loss_raw
    distill_loss = (distill_coeff if stage == 2 else 0.0) * kv_gate * distill_loss_raw

    loss = denoise_loss + proj_loss + distill_loss
```

工程优化（代码已实现）：
- 单 encoder 且 `repa` + `kv` 同时激活时，走联合路径，只做一次 encoder forward。
- 当两个辅助分支都关闭时，跳过 encoder forward，节省训练开销。

## Results (Current Snapshot)

| SiT-XL | FID  | sFID | IS     | pr    | Rc    |
| ------ | ---- | ---- | ------ | ----- | ----- |
| 400k   | 6.71 | 5.31 | 137.54 | 0.712 | 0.655 |
| 500k   | 6.57 | 5.38 | 140.30 | 0.706 | 0.663 |
| 650k   | 6.22 | 5.29 | 146.13 | 0.707 | 0.668 |
| 740k   | 5.70 | 5.18 | 149.59 | 0.711 | 0.670 |
| 860k   | 5.85 | 5.23 | 150.24 | 0.708 | 0.675 |
| 1M     | 7.00 | 5.59 | 140.73 | 0.692 | 0.674 |

结论写法建议：
- 前期加速成立：在远少于 baseline 7M 的训练步数下获得更优 FID。
- REPA early-stop 通常稳定有效；KV 分支后期是否保留应按质量-算力边际收益选择。

## Notes for Paper Framing

- 方法定位：operator-level routing prior（K/V injection）+ stage-wise release。
- 不夸大：避免“两个辅助损失必须全程同时开启”这类绝对表述。
- 负结果管理：SNR 加权、多层 K/V、复杂 MLP 投影若未稳定增益，可放附录而非主方法。
