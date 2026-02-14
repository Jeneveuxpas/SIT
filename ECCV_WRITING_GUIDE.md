# Attention Scaffolding: Bypassing the Cold Start of Diffusion Transformers via Encoder KV Injection

---
## Abstract

Diffusion Transformers train slowly because early in training, high-noise inputs often yield diffusion features with low spatial contrast; coupled with random initialization, this leads to high-entropy attention routing. Representation alignment (REPA) mitigates training challenges by aligning projections of noisy hidden states to clean encoder representations, yet recent evidence indicates that **spatial structure** rather than global semantic performance drives the gains. We propose **Attention Scaffolding**, a train-time mechanism that injects projections of pretrained encoder **K/V** into selected attention layers, providing an explicit spatial routing prior that bypasses this routing cold start. The scaffold is then removed by switching back to the model’s own K/V and distilling the induced routing structure for stability, with no change to inference and no extra tokens.

## 1 Introduction

Diffusion models have emerged as a powerful paradigm for high-fidelity image synthesis.  Recently, the community has moved from convolutional U‑Net backbones to Transformer architectures that process latent image patches.  Diffusion Transformers (DiTs) adopt Vision Transformer–style self‑attention over latent tokens and show strong scaling behaviour: increasing depth, width or token count steadily improves sample quality.  Building on this architecture, Scalable Interpolant Transformers (SiT) unify diffusion- and flow-style training in an interpolant framework and achieve further performance gains.  These trends suggest that Transformer-based generative models are an important direction for photorealistic synthesis.

However, DiT/SiT models are expensive to train.  Unlike convolutional networks, they lack built‑in spatial inductive biases, so the model must learn to organise image regions through self‑attention.  Early in training—especially for high-noise timesteps—token features exhibit low spatial contrast.  With random initialisation, the attention logits are weakly structured and route information almost uniformly, a high‑entropy regime we refer to as a **routing cold start**.  In practice, this cold start delays convergence, requiring longer training times and more compute to achieve high sample quality.  Accelerating training and improving efficiency is therefore critical to make such models practical for wider use.

Recent work has attempted to alleviate the cold start by leveraging pretrained vision encoders.  REPresentation Alignment (REPA) guides diffusion features towards clean-image representations from an external model and improves convergence.  Follow-up analysis reveals that REPA’s benefits correlate more with the spatial structure of an encoder’s patch tokens than with its global semantic accuracy.  iREPA introduces simple convolutional projections and spatial normalisation to emphasise spatial-structure transfer and obtains more consistent speedups.  Another line, HASTE, distils attention maps from a teacher but influences routing only indirectly via a loss term.  These methods demonstrate the value of transferring spatial information, yet they primarily operate at the feature level and do not provide a direct prior for the self-attention operator itself.

In this paper we close this gap by introducing **Attention Scaffolding**, a training mechanism that transfers spatial routing priors through the attention operator.  Our approach follows a two‑stage schedule.  During an initial warm‑up, we inject the keys and values from a frozen vision encoder into selected self‑attention layers of the diffusion model.  This scaffold encourages the student’s queries to attend according to the spatial structure learned by the encoder, thereby bootstrapping locality.  After the scaffold has established coherent routing, we restore the student’s native keys and values and continue training while distilling the scaffold‑induced routing behaviour.  The scaffold is used only during training; at inference, the model is identical to the base DiT/SiT with no additional tokens, modules, or runtime overhead.

**Contributions.**  Our work makes the following contributions:
- **Analysis of the routing cold start.** We identify and characterise the cold‑start problem in Transformer-based diffusion models and show how it limits training efficiency.
- **Operator‑level transfer via Attention Scaffolding.** We propose a simple, plug‑and‑play training mechanism that directly infuses spatial routing priors into the attention operator by temporarily injecting frozen encoder keys and values.    
- **Accelerated convergence and improved quality.** Through experiments on DiT and SiT backbones, we demonstrate that Attention Scaffolding reduces training time and improves sample quality compared to existing feature‑level and attention‑level guidance methods.
- **No inference overhead.** Our scaffold is only used during training; the resulting model preserves the inference speed and architecture of the base model.
## 相关工作

1. **REPA**：在 feature 空间对齐扩散模型中间表示与外部编码器表征，17.5× 加速 [REPA]。
2. **REG**：将编码器 CLS token entangle 进生成过程以提供全局语义指导 [REG]。与我们关注空间结构维度不同，REG 侧重语义维度的增强。
3. **iREPA**：发现空间结构比全局语义更能预测生成质量（|r| > 0.852 vs 0.26）[iREPA]。传递的应该是**空间结构**。
4. **HASTE**：发现 REPA 后期梯度与去噪目标冲突，提出 early-stop + attention map 蒸馏 [HASTE]。
5. **注意力蒸馏**：DeiT 通过 distillation token 实现注意力层面的知识传递 [DeiT]。
6. **K/V 操作**：风格迁移中替换 K/V 可无训练地传递空间和风格信息 [Style Injection]。



## 方法

### 1. Encoder K/V 捕获与投影

通过前向钩子从冻结的视觉编码器中提取特定层的 K/V，通过线性投影将维度映射到 SiT 的维度。对编码器架构透明，支持 DINOv2、ViT、SAM2 等。

### 2. 两阶段训练策略

**Stage 1 — 脚手架搭建**：用编码器 K/V 直接替换 SiT 的 K/V，模型的 Query 与外部 K/V 交互。从第一步即可利用正确的空间结构。

**为什么脚手架必须拆除？** 编码器的 K/V 来自干净图像，而扩散模型的 Query 来自不同噪声水平的潜变量——长时间维持这种分布不匹配会限制模型学习噪声级相关的注意力模式。消融验证：Stage 1 步数过长反而导致 FID 下降。

**Stage 2 — 脚手架拆除与蒸馏**：切回模型自有的 K/V，通过注意力输出蒸馏损失（MSE）保持空间结构一致性。投影层梯度 detach，只让 SiT attention 向编码器靠近。

### 3. 双层对齐：特征 + 注意力

将 K/V 注入与 REPA 结合：中间层（第 10 层）REPA 特征对齐 + 浅层（第 4 层）K/V 注入/蒸馏。两者互补——去掉 REPA: FID 12→~18；去掉 K/V: 回退到 REPA 基线。两者缺一不可。

### 4. 训练调度

HASTE [HASTE] 已证明 REPA 在训练后期与去噪目标冲突。我们验证了这一结论：关闭 REPA 后 FID 7.09→6.71。KV 蒸馏无此问题——关闭反而性能下降（6.71→7.25），可安全保留。

```
0 → stage1_steps:       Stage 1（K/V 注入 + REPA）
stage1_steps → t_repa:  Stage 2（K/V 蒸馏 + REPA）
t_repa → end:           Stage 2（仅 K/V 蒸馏，REPA 关闭）
```

### 总损失函数

$$\mathcal{L} = \mathcal{L}_{\text{denoise}} + \lambda_{\text{repa}} \cdot \mathbb{1}[n < \tau_{\text{repa}}] \cdot \mathcal{L}_{\text{REPA}} + \lambda_{\text{kv}} \cdot \mathbb{1}[\text{stage}=2] \cdot \mathcal{L}_{\text{distill}}$$

## 实验与分析

ImageNet 256×256，SiT-B/L/XL，DINOv2-B 编码器，对比 REPA 原论文数据和 SiT baseline。

- **训练效率**：SiT-XL/2 上 400K 步 FID≈6.71，超越 SiT baseline 7M 步 FID 8.3。
- **空间结构质量（核心亮点）**：sFID 在 100K 步达到 5.27 < REPA 4M 步的 5.73。这是 K/V 注入直接传递空间先验、iREPA 理论预测的最直接验证。
- **特征与注意力互补**：仅 REPA 或仅 K/V 都导致 FID 显著恶化，结合两者获得明显收益。
- **脚手架时长**：Stage 1 约 50K 步即可；过长则因分布不匹配性能下降。
- **REPA Early-Stop**：遵循 HASTE [HASTE] 的发现，250K 步关闭 REPA 后 FID 7.09→6.71；KV 蒸馏关闭则 6.71→7.25，保留至训练结束。

## 讨论与未来方向

本工作从 iREPA "空间结构最重要"的发现出发，提出直接注入 K/V 作为空间先验的注意力脚手架，与特征级 REPA 形成互补，采用 HASTE 提出的 early-stop 策略优化训练调度，实现训练加速和空间结构质量提升。

与 HASTE 的区别：HASTE 在 attention 外部蒸馏注意力图（CE loss），本质仍是间接引导；我们直接在 attention 内部注入 K/V，是更直接的空间结构传递路径。消融实验表明 KV 蒸馏可安全保留至训练结束，与 REPA 需要 early-stop 形成对比。

计算开销：编码器冻结不参与反传，额外开销有限；收敛加速带来的总训练时间节省远大于此。

未来工作：
- 更大分辨率和 text-to-image 任务；
- 不同编码器架构和层选择；
- 与其他表征增强方法结合探索。