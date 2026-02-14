# Attention Scaffolding: Bypassing the Cold Start of Diffusion Transformers via Encoder KV Injection

---
## Abstract

Diffusion Transformers train slowly because early in training, high-noise inputs often yield diffusion features with low spatial contrast; coupled with random initialization, this leads to high-entropy attention routing. Representation alignment (REPA) mitigates training challenges by aligning projections of noisy hidden states to clean encoder representations, yet recent evidence indicates that **spatial structure** rather than global semantic performance drives the gains. We propose **Attention Scaffolding**, a train-time mechanism that injects projections of pretrained encoder **K/V** into selected attention layers, providing an explicit spatial routing prior that bypasses this routing cold start. The scaffold is then removed by switching back to the model’s own K/V and distilling the induced routing structure for stability, with no change to inference and no extra tokens.

## 1 Introduction

Diffusion models have become a dominant paradigm for high-fidelity image synthesis, and a major recent shift is to replace U-Net backbones with Transformer architectures that operate on latent-image patches. **Diffusion Transformers (DiTs)** adopt ViT-style self-attention over latent tokens and exhibit strong scaling behavior: increasing depth/width or token count (and thus compute) consistently improves sample quality.  Building on this backbone, **Scalable Interpolant Transformers (SiT)** unify diffusion- and flow-style training under an interpolant framework and further improve performance while retaining the same Transformer structure. 

Despite their scalability, DiT/SiT models are often **slow to train**. A key reason is that spatial organization must be _discovered_ through attention routing rather than being baked in by convolutional inductive biases. Early in optimization—especially at high-noise timesteps—token features typically exhibit **low spatial contrast**, and with randomly initialized projections the attention logits tend to be weakly structured, yielding **nearly uniform, high-entropy routing**. In this regime, the model struggles to form coherent locality and intermediate representations, delaying the emergence of stable token-to-token information pathways. We refer to this phenomenon as a **routing cold start**: the attention operator initially lacks an effective spatial prior, making structured routing difficult to bootstrap.

A recent line of work alleviates this burden by importing strong visual representations from pretrained encoders. **REPresentation Alignment (REPA)** accelerates optimization and improves final generation quality by aligning intermediate diffusion features (under noisy inputs) to clean-image representations produced by an external pretrained vision model.  Importantly, follow-up analysis shows that REPA’s gains correlate **weakly** with an encoder’s _global semantic strength_ (e.g., ImageNet accuracy), but **strongly** with the _spatial structure_ of its patch tokens—captured by pairwise patch-to-patch similarity geometry that reflects layout and grouping.  This evidence is further strengthened by **iREPA**, which makes minimal, targeted changes (convolutional projection and spatial normalization) to emphasize spatial-structure transfer and yields more consistent convergence improvements. 

However, REPA/iREPA remain fundamentally **feature-level** alignment methods: they constrain _what_ each token encodes, but do not directly shape the **routing operator**—the token-to-token attention patterns that determine _how_ spatial relations are propagated across layers. **HASTE** moves toward attention-level guidance via attention-map distillation, yet it remains an **indirect, loss-level constraint** and does not provide an explicit **K/V routing prior** to bypass the early-stage routing cold start.

Motivated by this gap, we propose **Attention Scaffolding**, an **operator-level** training mechanism that transfers spatial routing priors through attention itself. Our approach follows a simple two-stage schedule: we first inject a frozen encoder’s **K/V** into selected self-attention layers to bootstrap structured routing, and then restore the student’s native **K/V** while transferring the scaffold-induced routing behavior into the student. The scaffold is used only during training—at inference time, the model is identical to the base DiT/SiT, with **no additional tokens, modules, or runtime overhead**.

---

## **Contributions**

1. **Routing cold start perspective.** We identify unstable, high-entropy attention routing under high-noise/early-stage conditions as a concrete bottleneck for DiT/SiT training efficiency.
2. **Attention Scaffolding: operator-level spatial prior transfer.** We introduce a two-stage mechanism that (i) injects frozen encoder K/V to bootstrap structured routing and (ii) repatriates the induced organization into the student via routing-preserving distillation.
3. **Complementary to feature alignment, inference unchanged.** Our approach complements REPA-style feature-level alignment by directly transferring routing structure. On ImageNet 256×256, SiT-XL/2 with Attention Scaffolding reaches FID [**TODO**] at [**TODO**] steps. The spatial-structure metric sFID reaches [**TODO**] at only [**TODO**] steps, surpassing REPA's 5.73 at 4M steps.

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