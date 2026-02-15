# Attention Scaffolding: Bypassing Routing Cold Start in Diffusion Transformers via Encoder K/V Injection

---
## Abstract

Diffusion Transformers suffer from a routing cold start: at high noise and random initialization, Q-K interactions are weakly structured, slowing convergence. We propose **Attention Scaffolding**, a training-only mechanism that injects pretrained encoder keys/values into selected DiT/SiT attention layers during early training, then releases the scaffold and trains with native K/V. We combine this operator-level prior with feature-level REPA under a stage-wise schedule. Empirically, REPA early-stop is consistently beneficial, while K/V distillation mainly contributes in the early-to-mid regime and can be retained or stopped based on quality-compute tradeoff. The final model has no inference-time overhead.

## 1 Introduction

DiT/SiT models scale well but are expensive to train. A core bottleneck is early routing formation: without strong spatial inductive bias, the model needs many updates before attention develops stable locality. Existing acceleration methods mostly align features (e.g., REPA/iREPA) or distill attention targets externally. We focus on a more direct path: injecting encoder K/V into the student attention operator to provide a spatial routing scaffold at the beginning of training.

Our central claim is pragmatic:
- **Early phase:** operator-level K/V scaffold improves routing bootstrap efficiency.
- **Middle/late phase:** feature-level alignment (REPA) should be terminated early (as in HASTE), and K/V objectives should be scheduled by observed marginal gain rather than kept by default.

This framing matches observed behavior in our SiT experiments: strong early acceleration, but late-stage quality depends on schedule and checkpoint selection.

## 相关工作

1. **REPA**：在特征空间对齐扩散表示与外部编码器表示，显著提升前期收敛 [REPA]。
2. **iREPA**：指出空间结构相关信号比全局语义更能预测生成质量 [iREPA]。
3. **HASTE**：提出 holistic alignment 的阶段终止（early-stop），缓解后期与去噪目标冲突 [HASTE]。
4. **注意力蒸馏/注意力干预**：通过注意力层知识迁移提升训练稳定性 [DeiT, etc.]。
5. **K/V 操作**：在视觉生成与编辑中，K/V 级别操作可直接影响空间与风格路由。

## 方法

### 1. Encoder K/V 捕获与投影

从冻结视觉编码器（如 DINOv2）提取指定层 K/V，经投影映射到 SiT hidden dim。投影可为 `linear` 或 `mlp`，默认优先 `linear` 作为强基线。

### 2. 两阶段 Attention Scaffolding

**Stage 1（routing bootstrap）**  
在指定 SiT 层用 encoder K/V 替换 student K/V，让 student query 在训练初期直接接入较稳定的空间路由先验。

**Stage 2（scaffold release）**  
恢复 student 自有 K/V，仅保留（可选）distillation loss 约束路由连续性，避免 Stage 1 -> Stage 2 硬切换带来的不稳定。

关键点：Stage 1 不宜过长。encoder K/V 来自 clean image 分布，student 输入来自噪声潜变量，长期强绑定会形成分布错配。

### 3. 与 REPA 的协同

我们采用“特征 + 注意力”双通道监督，但作用不同：
- **REPA**：主导语义/表征对齐，前中期有效，后期可能与去噪目标冲突，建议 early-stop。
- **K/V distill**：主导 routing 迁移，前期收益明显；后期收益常趋于饱和，可按性价比决定是否继续。

因此不是“两个损失必须全程同时开启”，而是“**分工互补 + 分阶段调度**”。

### 4. 训练调度（建议写法）

```
0 -> stage1_steps:         Stage 1 (K/V 注入 + REPA)
stage1_steps -> tau_repa:  Stage 2 (K/V distill + REPA)
tau_repa -> tau_kv:        Stage 2 (K/V distill only)
tau_kv -> end:             Denoising only (optional)
```

其中 `tau_repa` 固定早停（如 250k）通常稳定；`tau_kv` 建议由验证集边际收益或梯度监控决定。

### 总损失函数

$$
\mathcal{L}
=
\mathcal{L}_{\text{denoise}}
+
\lambda_{\text{repa}} \, g_{\text{repa}}(n)\, \mathcal{L}_{\text{REPA}}
+
\lambda_{\text{kv}} \, g_{\text{kv}}(n)\, \mathcal{L}_{\text{distill}}
$$

其中：
- $g_{\text{repa}}(n)=\mathbf{1}[n<\tau_{\text{repa}}]$
- $g_{\text{kv}}(n)$ 可取 hard-stop、fade、或常数 1（不停止）

实现上，当 `g_repa=0` 且 `g_kv=0` 时可跳过 encoder 前向，降低训练开销。

## 实验叙事（基于当前结果）

### 1. 先给稳定事实

- SiT-XL 在 400k/500k 已明显优于原始长程 baseline（训练效率提升成立）。
- REPA early-stop 带来稳定正收益（与当前实验一致）。
- 去掉 REPA 会显著退化（说明特征对齐仍是主干增益来源）。

### 2. 再给 KV 的真实定位

- KV scaffold 在前期有效，尤其改善 early routing。
- KV 在后期是否继续保留，收益依赖具体配置；你当前结果显示其后期增益偏小、对 compute 有额外成本。
- 因此论文结论应写成：**KV is an early accelerator, optionally retained in late stage.**

### 3. 对“后期波动”给合理解释

- FID 非严格单调，checkpoint 间存在统计噪声与训练轨迹波动。
- 应报告“best-in-window”而非只报最后一步；建议 500k 以后每 25k/50k 密集评估。

### 4. 当前不建议强推的点

- SNR-weighted attn/kv 在你现有实验中未显示稳定优势，不作为主方法结论。
- 多层 K/V 与复杂 MLP 投影当前未超过线性基线，不作为主卖点，可放附录负结果。

## 讨论与写作建议

### 你的方法一句话版本

We bypass DiT routing cold start with a training-time K/V scaffold, then release it stage-wise while applying REPA early-stop to avoid late objective conflict.

### 和 HASTE 的关系（建议措辞）

- HASTE：强调 alignment should stop when conflicting with denoising.
- 你：在此基础上把“对齐对象”下沉到 attention operator（K/V 注入），并给出可执行的 staged schedule。

### 未来方向

- 自适应 `tau_kv`（基于梯度相似度或验证指标触发）
- 更大规模训练（>1M）下的 schedule 稳定性
- text-to-image 与多编码器迁移
