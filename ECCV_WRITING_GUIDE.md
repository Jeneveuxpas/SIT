# Attention Scaffolding: Bypassing the Cold Start of Diffusion Transformers via Encoder KV Injection

---
## Abstract

Diffusion Transformers train slowly because early in training, high-noise inputs often yield diffusion features with low spatial contrast; coupled with random initialization, this leads to high-entropy attention routing. Representation alignment (REPA) mitigates training challenges by aligning projections of noisy hidden states to clean encoder representations, yet recent evidence indicates that **spatial structure** rather than global semantic performance drives the gains. We propose **Attention Scaffolding**, a train-time mechanism that injects pretrained encoder **K/V** into selected attention layers, providing an explicit spatial routing prior that bypasses this routing cold start. The scaffold is then removed by switching back to the model’s own K/V and distilling the induced routing structure for stability, with no change to inference and no extra tokens.

**中文参考**：扩散 Transformer 训练缓慢的关键在于：去噪早期输入高噪，token 表征的空间对比度很低，导致 self-attention 路由高熵、难以稳定形成空间结构。REPA 通过将噪声隐状态对齐到编码器干净表征来加速训练，但最新证据表明带来生成收益的核心是 patch 间空间结构而非全局语义。我们提出 Attention Scaffolding：训练早期注入编码器 K/V 提供空间路由先验以绕过冷启动，随后切回自身 K/V 并蒸馏固化，推理不加 token、无额外依赖。SiT-XL/2 在 [**TODO**] 步达到 FID [**TODO**]；sFID 在 [**TODO**] 步即超越 REPA 4M 步。

## 引言与动机

近期研究表明，扩散变换器（DiT/SiT）在生成质量上达到最先进水平，但训练速度极慢。REPA 通过将扩散模型的中间表示与外部视觉编码器 feature 对齐，显著加速训练——将 SiT 的训练步数从 7M 减少到 400K（加速 17.5×）[REPA]。然而，REPA 究竟在传递什么信息来加速训练？是全局语义，还是空间结构？

iREPA 对这一问题进行了深入分析。实验证明，衡量 patch 之间两两相似性的空间结构指标与生成 FID 的相关性显著更高（Pearson |r| > 0.852），而 ImageNet-1K 准确率的相关性仅为 0.26 [iREPA]。甚至 ImageNet 准确率仅 24.1% 的 SAM2-S 在 REPA 框架下也能优于准确率高出 60% 的编码器 [iREPA]。这一系列结果强烈暗示：加速训练的关键在于传递**空间结构**，而非全局语义。问题由此变为：**如何更有效地传递空间结构？**

现有方法对空间结构的传递仍然是间接的。REPA/iREPA 通过 feature 层面的 loss 间接引导；HASTE 增加了 attention map 蒸馏，但本质仍是在 attention 外部施加损失 [HASTE]。这些方法都依赖"损失→梯度→权重更新→注意力变化"的间接路径。训练初期，随机初始化的 Q/K/V 产生近乎均匀的注意力分布，模型需要大量迭代才能建立有意义的空间注意力模式——**冷启动是训练效率的主要瓶颈**。

## 关键洞察：K/V 是空间先验的直接载体

在自注意力中，Key 编码"哪些位置之间应建立关联"，Value 承载"从被关注位置传递的信息"。预训练编码器处理干净图像后的 K/V 天然包含完整的空间关系先验。风格迁移文献已证实：替换自注意力的 K/V 可以在保持内容布局的同时转移局部纹理 [Style Injection]，说明 K/V 的确包含空间结构信息。



基于此，我们提出 **Attention Scaffolding**（注意力脚手架）——直接将预训练编码器的 K/V 注入扩散 Transformer 的 attention 层，让模型从第一步就跳过冷启动阶段。

### Contributions

1. 提出 attention-level 空间结构传递：直接注入 encoder K/V 作为脚手架跳过注意力冷启动，作为对 feature-level 对齐（REPA）的互补维度。
2. 设计两阶段训练策略（Stage 1 直接注入 → Stage 2 蒸馏过渡），并结合 HASTE 提出的 early-stop 策略 [HASTE]，形成完整的训练调度方案。消融实验验证了各组件的必要性。
3. 在 SiT-XL/2 上 400K 步达到 FID 6.71；衡量空间结构质量的 sFID 在 100K 步即达到 5.27，优于 REPA 在 4M 步的 5.73，直接验证了 attention-level 空间结构传递的有效性。

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