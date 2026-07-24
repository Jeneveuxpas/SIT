# AttnScaf 方法与实验事实记录

更新日期：2026-07-22

本文档只记录当前代码实现、训练协议、已有实验数字和待核实事项，不包含论文故事、创新性判断或结果解释。新对话可直接引用本文件作为背景。

## 0. 新对话事实摘要（2026-07-22）

以下是当前最重要且已记录的实验事实：

1. **完整hidden-state replacement没有带来改善。** 在SiT block 8进行30K完整替换、随后恢复原生SiT并训练到100K时，FID为53.94；当前Vanilla SiT参考为39.20。
2. **带clean reference的30K oracle诊断中，完整hidden replacement与K/V scaffold差异极大。** 保持外部clean-image representation在每个采样步参与前向时，block-8 hidden replacement的FID为212.53，而block-8 K/V scaffold的FID为3.91。该oracle设置使用了正常推理不可获得的对应clean image，不能与标准无DINO生成FID作同口径比较。
3. **撤除外部表征后的block-8 K/V结果形成清晰阶梯。** AttnScaf-only为22.45；加入attention-output consistency后为19.08；再加入5K smooth withdrawal后为17.22（均为100K、无REPA）。
4. **已有block-4 standalone结果仍有效，但必须与block-8结果分开标注。** Block 4上，AttnScaf-only为23.61，+consistency为18.93，+consistency+smooth为18.06。
5. **当前正在运行的新组合实验**将K/V scaffold放在SiT block 4，将REPA alignment放在SiT block 8，并使用consistency、5K smooth withdrawal和SpatialNorm。该实验结果尚未写入本文档。

当前事实不支持“完整hidden replacement带来小幅提升”这一旧假设。当前结果只支持：完整替换失败，而保留native noisy query与residual路径的K/V接口明显更有效。

## 1. 实验基础设置

### 1.1 模型与数据

- 生成模型：SiT-XL/2。
- 数据集：ImageNet 256×256，1000类。
- 视觉编码器：冻结的 DINOv2-B。
- DINO输入：与训练latent对应的清晰图像 `x_0`，经过DINO预处理。
- 默认DINO K/V来源：`enc-layer-indices: "12"`，即DINOv2-B第12个block的attention Q/K/V activations。
- 默认SiT注入位置：`sit-layer-indices: "4"`，为1-based的SiT第4个block。
- SiT block 8用于与官方REPA位置对齐的matched-depth实验；SiT block 10也用于层位置和其他接口消融。
- 训练batch size：256。
- 优化器学习率：`1e-4`。
- EMA decay：`0.9999`。
- 当前主结果均为seed 0，尚无多seed均值与方差。

### 1.2 当前评估脚本默认值

`eval_ckpts.sh`当前默认：

- 50,000张样本；
- CFG scale = 1.0；
- SDE sampler；
- 250 sampling steps；
- seed 0；
- batch size 256。

注意：当前`eval_ckpts.sh`默认`VAE=mse`，而`launch.sh`默认`VAE=ema`。已有截图中的部分结果文件名为`vaeema`。整理正式表格前，应根据每个结果文件名核实所有行使用的VAE是否完全一致。

## 2. K/V AttnScaf实现

### 2.1 注入对象

方法注入的是DINO对清晰图像前向产生的K/V activations，不是DINO的`W_K/W_V`权重矩阵。

设SiT指定block的原生Q/K/V为：

\[
Q_S,K_S,V_S=\operatorname{QKV}_{\mathrm{SiT}}(h_l),
\]

DINO指定block产生：

\[
K_E(x_0),V_E(x_0).
\]

两个独立线性层将DINO K/V投影到SiT维度：

\[
K_T=P_K(K_E(x_0)),\qquad V_T=P_V(V_E(x_0)).
\]

默认K/V scaffold计算为：

\[
O_T=\operatorname{Attn}(Q_S,K_T,V_T).
\]

### 2.2 默认投影与归一化

- `kv-replace-mode: kv`；
- `kv-proj-type: linear`；
- K和V使用两个独立的无bias线性映射；
- `qk-norm: false`；
- `kv-norm-type: none`。

因此，当前主表中的K/V AttnScaf结果不包含K/V normalization。

### 2.3 硬切换协议

标准硬切换设置：

- 0–30K：去噪主前向使用`Attn(Q_S, K_T, V_T)`；
- 30K以后：恢复原生`Attn(Q_S, K_S, V_S)`。

前30K中：

- SiT的Q路径被使用并获得梯度；
- SiT原生K/V虽被计算，但在硬切换版本中不参与attention output，因此对应参数基本没有有效梯度；
- DINO K/V投影`P_K/P_V`获得梯度；
- SiT attention output projection、residual、MLP、前缀blocks及后续blocks正常参与训练。

30K以后：

- `P_K/P_V`默认冻结；
- 主前向使用SiT原生Q/K/V；
- 原生K/V开始从denoising loss和可选的consistency loss获得梯度。

### 2.4 Smooth 5K协议

`transition-steps: 5000`时：

- 0–25K：纯K/V scaffold；
- 25–30K：在attention聚合输出层面做cosine blending；
- 30K以后：纯原生self-attention。

渐变对象是attention output，不是K/V张量：

\[
O(\alpha)=(1-\alpha)O_T+\alpha O_S,
\]

其中：

\[
O_S=\operatorname{Attn}(Q_S,K_S,V_S).
\]

在25–30K渐变窗口内，原生K/V路径参与输出，因此能够获得梯度。

### 2.5 Scaffold consistency

默认`align-mode: attn_mse`比较：

\[
O_T=\operatorname{Attn}(Q_S,K_T,V_T),
\]

与：

\[
O_S=\operatorname{Attn}(Q_S,K_S,V_S).
\]

损失为：

\[
\mathcal L_{\mathrm{sc}}=\|O_S-\operatorname{sg}(O_T)\|_2^2.
\]

具体实现事实：

- 比较的是多头attention聚合结果；
- 比较位置在共享attention output projection之前；
- 不是attention map loss；
- 不是K/V直接回归；
- consistency系数在0–30K为0，只在切换到原生attention后启用；
- 默认`distill-coeff = 2.0`；
- teacher使用当前SiT query和冻结的DINO K/V投影在线计算；
- 默认`kv-stop-step: -1`时，DINO在30K以后仍每步前向，用于产生teacher；
- DINO K/V在30K以后不再进入denoising主前向。

### 2.6 AttnScaf-only设置

`attnscaf`（FID 23.61）协议：

- 0–30K使用K/V scaffold；
- 30K硬切换到SiT原生K/V；
- `distill-coeff = 0`；
- 不使用REPA；
- `kv-stop-step = 30000`，所以30K以后不再运行DINO分支。

### 2.7 Stop100K/Fade20K设置

该名字描述的是consistency loss停止日程，不是scaffold撤除时间：

- 0–30K：K/V scaffold；
- 30–100K：完整consistency系数；
- 100–120K：consistency系数衰减至0；
- 120K以后：仅denoising loss，DINO分支关闭。

scaffold仍然在30K退出主前向。

### 2.8 推理

- 推理时不加载DINO；
- 不使用外部K/V；
- 使用SiT原生Q/K/V；
- 不改变生成变量或采样器。

### 2.9 带clean reference的oracle诊断

仓库中的`scripts/sample_hidden_replacement_oracle.py`支持对30K stage-1 checkpoint进行hidden或K/V oracle采样：

- 从随机噪声开始正常执行扩散采样；
- 预先对一张reference clean image运行冻结视觉编码器；
- 在每个采样步复用同一份clean representation或DINO K/V；
- hidden接口将投影后的clean representation用作指定位置的完整hidden state；
- K/V接口保留SiT原生query，并使用reference产生的K/V；
- class label由命令行显式提供，进行同类别实验时应与reference image类别一致。

这不是标准生成协议，因为它在采样期间使用对应clean image。其用途是诊断下游SiT能否利用不同接口提供的privileged clean-image信息，不应把oracle FID直接当作无外部信息的生成性能。

## 3. REPA与SpatialNorm设置

### 3.1 REPA目标

- 官方REPA参考结果使用SiT block 8进行alignment（由`encoder-depth: 8`控制）；
- 本报告较早的AttnScaf+REPA组合实验多数使用`encoder-depth: 10`；
- 新的分层组合配置使用AttnScaf block 4与REPA block 8，必须与旧的REPA block 10组合结果分开标注；
- DINO target为清晰图像的最终归一化patch tokens：`x_norm_patchtokens`；
- 标准REPA使用cosine alignment；
- 当前最佳组合结果使用token-wise MLP projection head。

### 3.2 SpatialNorm

`spnorm-method: zscore`作用于REPA的DINO patch target，不作用于K/V。沿空间token维进行：

\[
\hat z_{btd}=\frac{z_{btd}-\alpha\mu_{bd}}{\sigma_{bd}+\epsilon},
\qquad \alpha=0.6.
\]

随后进行token-wise cosine alignment。

表格中的`REPA`和`REPA+SpatialNorm`应视为不同baseline。

## 4. 其他接口实现

### 4.1 Feature Residual / Attention-output replacement

当前Feature Residual使用DINO最终归一化patch tokens：

\[
F_E(x_0)=\texttt{x\_norm\_patchtokens}.
\]

经无bias线性层映射到SiT hidden dimension：

\[
R_T=P_R(F_E(x_0)).
\]

在scaffold窗口中，替换指定SiT block的attention branch output：

\[
A_l^{S}\leftarrow R_T.
\]

原始hidden state和residual addition仍保留：

\[
h'_l=h_l+g_l^{\mathrm{attn}}\odot R_T.
\]

之后的MLP branch保持原生计算。切换后恢复原生attention output，并以MSE对齐映射后的DINO feature。

### 4.2 Full Hidden Replacement

在scaffold窗口执行：

\[
h_l\leftarrow P_H(F_E(x_0)).
\]

该实现直接返回映射后的DINO feature，因此替换位置之前的SiT路径及被替换block自身不通过该路径获得denoising gradient。切换后恢复原生block，并以MSE比较原生block输出与冻结的DINO映射结果。

### 4.3 Patch-shuffle control

Patch shuffle发生在DINO前向之前：

- 将DINO预处理后的输入图像切成16×16个patch；
- 每个样本独立随机打乱图像patch；
- 再把打乱后的图像输入DINO并提取K/V。

该实验不是在DINO输出后重排K/V token pair。

## 5. 已整理的定量结果

以下结果来自截至2026-07-22的汇总表。FID和sFID越低越好，IS、Precision和Recall越高越好。

### 5.1 K/V AttnScaf + consistency + REPA + SpatialNorm

| 配置 | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Hard transition | 100K | 11.95 | 6.04 | 95.71 | 0.70 | 0.61 |
| Hard transition | 200K | 7.94 | 4.99 | 123.50 | 0.71 | 0.64 |
| Hard transition | 400K | 6.63 | 5.13 | 137.03 | 0.71 | 0.67 |
| Smooth 5K | 100K | 11.30 | 5.85 | 98.47 | 0.70 | 0.61 |

备注：Smooth combined的400K评估在当前记录中尚无结果。

### 5.2 Standalone consistency系数消融（无REPA）

| `distill-coeff` | Iter | FID | sFID | IS | Precision | Recall |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100K | 20.09 | 5.38 | 65.48 | 0.66 | 0.58 |
| 1 | 400K | 8.02 | 5.10 | 125.09 | 0.71 | 0.64 |
| 2 | 100K | 18.93 | 5.59 | 69.12 | 0.67 | 0.58 |
| 4 | 100K | 18.63 | 5.50 | 69.84 | 0.67 | 0.59 |
| 4 | 400K | 8.13 | 5.19 | 124.71 | 0.70 | 0.65 |

备注：`lambda=2, FID=18.93`对应SiT block 4；`FID=19.08`对应SiT block 8。两者是不同注入深度，不再视为同一配置的数字冲突。

### 5.3 Standalone smooth、层位置和scaffold时长

| 配置 | SiT layer | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Hard 30K + consistency | 4 | 100K | 18.93 | 5.59 | 69.12 | 0.67 | 0.58 |
| Smooth 5K + consistency | 4 | 100K | 18.06 | 5.38 | 70.32 | 0.67 | 0.59 |
| Smooth 5K + consistency | 4 | 200K | 10.04 | 5.11 | 107.89 | 0.71 | 0.62 |
| Smooth 5K + consistency | 4 | 400K | 7.79 | 5.20 | 128.05 | 0.71 | 0.65 |
| Hard 25K + consistency | 4 | 100K | 18.77 | 5.50 | 70.26 | 0.66 | 0.60 |
| Hard 30K, no consistency | 8 | 100K | 22.45 | 5.38 | 57.45 | 0.65 | 0.59 |
| Hard 30K + consistency | 8 | 100K | 19.08 | 5.51 | 68.45 | 0.66 | 0.59 |
| Smooth 5K + consistency | 8 | 100K | 17.22 | 5.38 | 72.65 | 0.67 | 0.59 |
| Smooth 5K + consistency | 10 | 100K | 19.03 | 5.60 | 67.58 | 0.66 | 0.60 |

### 5.4 Consistency stop消融（无REPA）

| 配置 | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| stop100K, fade20K | 200K | 10.89 | 5.10 | 103.57 | 0.70 | 0.62 |
| stop100K, fade20K | 400K | 8.49 | 5.11 | 120.78 | 0.71 | 0.64 |

### 5.5 Feature Residual与Hidden Replacement

| 配置 | SiT layer | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Feature Residual only | 10 | 100K | 35.29 | 5.52 | 37.41 | 0.57 | 0.59 |
| Feature Residual + consistency | 10 | 100K | 17.65 | 6.25 | 74.38 | 0.67 | 0.60 |
| Feature Residual + consistency + REPA | 10 | 100K | 16.81 | 6.21 | 79.11 | 0.66 | 0.61 |
| Feature Residual + consistency + REPA | 4 | 100K | 15.00 | 5.74 | 84.29 | 0.67 | 0.60 |
| Full Hidden Replacement only | 8 | 100K | 53.94 | 8.05 | 24.36 | 0.45 | 0.58 |
| Full Hidden Replacement + consistency | 10 | 100K | 21.74 | 6.09 | 66.54 | 0.62 | 0.63 |

`Full Hidden Replacement only, layer 8`的协议为：前30K使用完整hidden replacement，之后撤除外部路径并以原生SiT继续训练至100K；不使用REPA或consistency。该结果不能被表述为相对Vanilla SiT的提升。

其他局部接口结果：

| 配置 | SiT layer | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Attention-output + consistency | 4 | 100K | 27.53 | 5.47 | 50.05 | 0.61 | 0.61 |
| Attention-output + consistency | 10 | 100K | 26.76 | 5.80 | 52.40 | 0.61 | 0.60 |
| Final-feature K/V + consistency | 未标注 | 100K | 23.35 | 5.70 | 59.70 | 0.63 | 0.60 |

这些接口的teacher定义与主K/V接口不同，不能只根据名称将其视为完全matched的替换位置消融；正式写表前仍需按checkpoint args核对。

### 5.6 Baselines与主要组合

| 配置 | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla SiT | 100K | 39.20 | — | — | — | — |
| REPA | 100K | 19.40 | 6.06 | 67.40 | 0.64 | 0.61 |
| iREPA | 100K | 16.90 | 6.26 | 77.92 | 0.66 | 0.61 |
| iREPA | 400K | 7.65 | 4.97 | 126.55 | 0.71 | 0.65 |
| iREPA | 600K | 6.91 | 4.98 | 134.67 | 0.71 | 0.66 |
| iREPA | 800K | 6.62 | 5.05 | 139.38 | 0.71 | 0.67 |
| REPA + SpatialNorm | 100K | 16.89 | 6.07 | 76.58 | 0.66 | 0.61 |
| REPA + SpatialNorm | 400K | 7.81 | 5.09 | 125.54 | 0.70 | 0.65 |
| AttnScaf only | 100K | 23.61 | 5.39 | 56.21 | 0.64 | 0.59 |
| AttnScaf only（block 8） | 100K | 22.45 | 5.38 | 57.45 | 0.65 | 0.59 |
| AttnScaf + consistency | 100K | 18.93 | 5.59 | 69.12 | 0.67 | 0.58 |
| AttnScaf + consistency（block 8） | 100K | 19.08 | 5.51 | 68.45 | 0.66 | 0.59 |
| AttnScaf + consistency + Smooth 5K（block 8） | 100K | 17.22 | 5.38 | 72.65 | 0.67 | 0.59 |
| AttnScaf + REPA + SpatialNorm（无consistency） | 100K | 12.08 | 5.98 | 94.66 | 0.70 | 0.61 |
| AttnScaf + REPA + SpatialNorm（无consistency） | 400K | 6.85 | 5.13 | 134.24 | 0.71 | 0.66 |
| AttnScaf + consistency + REPA + SpatialNorm | 100K | 11.95 | 6.04 | 95.71 | 0.70 | 0.61 |
| AttnScaf + consistency + REPA + SpatialNorm | 400K | 6.63 | 5.13 | 137.03 | 0.71 | 0.67 |

### 5.7 REPA alignment head和SpatialNorm消融

K/V mapping在这些实验中保持linear；表内MLP/Conv指REPA alignment head。

| SpatialNorm | REPA head | Iter | FID | sFID | IS | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 是 | MLP | 100K | 11.95 | 6.04 | 95.71 | 0.70 | 0.61 |
| 是 | MLP | 200K | 7.94 | 4.99 | 123.50 | 0.71 | 0.64 |
| 是 | MLP | 400K | 6.63 | 5.13 | 137.03 | 0.71 | 0.67 |
| 是 | Conv 3×3 | 100K | 12.41 | 5.34 | 92.12 | 0.70 | 0.61 |
| 是 | Conv 3×3 | 200K | 8.60 | 5.06 | 117.11 | 0.71 | 0.64 |
| 是 | Conv 3×3 | 400K | 7.21 | 5.18 | 130.21 | 0.71 | 0.66 |
| 否 | MLP | 100K | 12.08 | 5.78 | 94.87 | 0.70 | 0.61 |
| 否 | MLP | 200K | 8.04 | 5.05 | 123.68 | 0.71 | 0.65 |
| 否 | MLP | 400K | 6.77 | 5.15 | 136.40 | 0.71 | 0.66 |
| 否 | Conv 3×3 | 100K | 12.15 | 5.33 | 95.28 | 0.70 | 0.62 |
| 否 | Conv 3×3 | 200K | 8.47 | 5.11 | 118.24 | 0.70 | 0.64 |
| 否 | Conv 3×3 | 400K | 7.08 | 5.24 | 132.89 | 0.70 | 0.67 |

### 5.8 K-only、V-only、QKV结果

当前汇总表记录：

| Replacement与协议标签 | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| V only + REPA | 100K | 15.75 | 6.10 | 80.63 | 0.67 | 0.60 |
| K only + REPA | 100K | 12.55 | 5.88 | 92.69 | 0.69 | 0.60 |
| QKV + REPA | 100K | 14.56 | 5.82 | 84.76 | 0.68 | 0.61 |
| QKV + consistency + Smooth | 100K | 20.96 | 5.92 | 61.19 | 0.655 | 0.595 |
| V only（standalone） | 100K | 27.49 | 5.86 | 50.55 | 0.61 | 0.60 |

协议核查状态：

- 现有K-only/V-only配置文件为REPA + SpatialNorm + MLP head + consistency系数2；
- 旧`qkv.yaml`使用不同REPA head、不同loss和不同系数；
- 另有新的QKV standalone配置，不带REPA；
- 14.56究竟对应哪份checkpoint args尚未在本地确认。

因此，在核实原始checkpoint配置前，这三行不能视为完全matched的Q/K/V接口消融。主KV对应的完整matched参考数为11.95，但也需与最终确认的QKV协议一起整理。

### 5.9 Patch-shuffle

| 配置 | Iter | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Shuffled-image AttnScaf only | 100K | 32.25 | 6.29 | 43.20 | 0.58 | 0.60 |
| Shuffled-image AttnScaf + consistency | 100K | 28.89 | 6.42 | 48.26 | 0.59 | 0.61 |

对应未shuffle参考：

- AttnScaf only：23.61；
- AttnScaf + consistency：18.93；
- Vanilla SiT：39.20。

### 5.10 单独K/V norm试验

先前记录过一项`AttnScaf + consistency + K/V norm`的100K FID为19.28，其余指标和精确配置未记录在最新汇总表中。当前主结果均使用`kv-norm-type: none`。

### 5.11 30K clean-reference oracle诊断

| Scaffold interface | SiT layer | Checkpoint | FID | sFID | IS | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| K/V AttnScaf | 8 | 30K | 3.91 | 5.04 | 59.31 | 0.71 | 0.70 |
| Full Hidden Replacement | 8 | 30K | 212.53 | 125.63 | 2.89 | 0.03 | 0.027 |

两行均标记为在sampling期间保持对应clean-image DINO信息开启的oracle结果。它们可用于同一oracle协议内比较接口是否能利用privileged clean information，但不能与标准REPA、Vanilla SiT或撤除DINO后的AttnScaf FID直接作公平性能比较。

当前仓库中的单图oracle脚本可以验证样例级行为，但未发现其直接生成50K paired-reference FID批次的实现。因此，上述两行的批量评估入口、reference与生成样本的配对方式、类别匹配方式以及VAE设置仍需在正式论文表格前核实。

## 6. 当前结果之间的协议对应关系

- `18.06 / 10.04 / 7.79`：standalone、无REPA、Smooth 5K、SiT layer 4、consistency系数2。
- `17.22 @ 100K`：standalone、无REPA、Smooth 5K、SiT layer 8、consistency系数2。
- `19.08 @ 100K`：standalone、无REPA、硬切换、SiT layer 8、consistency系数2。
- `22.45 @ 100K`：AttnScaf-only、硬切换、无consistency、无REPA、SiT layer 8。
- `53.94 @ 100K`：Full Hidden Replacement-only、前30K在SiT layer 8完整替换、之后撤除，且无consistency与REPA。
- `3.91 / 212.53 @ 30K`：分别为block-8 K/V与Full Hidden Replacement在clean-reference信息持续开启时的oracle诊断结果，不是标准无DINO推理。
- `11.95 / 7.94 / 6.63`：AttnScaf + consistency + REPA + SpatialNorm、硬切换、SiT layer 4。
- `11.30 @ 100K`：AttnScaf + consistency + REPA + SpatialNorm、Smooth 5K、SiT layer 4。
- `23.61 @ 100K`：AttnScaf-only、硬切换、无consistency、无REPA，DINO在30K后完全关闭。
- `12.08 / 6.85`：AttnScaf + REPA + SpatialNorm，无consistency。
- `12.08 / 8.04 / 6.77`：AttnScaf + consistency + REPA，无SpatialNorm。该组100K FID与上一组恰好同为12.08，但协议不同。

## 7. 尚未得到或尚未完全核实的实验

- **AttnScaf block 4 + consistency + Smooth 5K + REPA block 8 + SpatialNorm**：配置已建立并正在/准备运行，100K结果尚未记录。
- Smooth combined 400K结果：当前正在/准备评估，尚未记录。
- AttnScaf-only 400K：尚未记录。
- Transient REPA（例如只开0–30K）：尚未记录。
- Random-init encoder K/V control：未记录。
- DINO noisy-input `K/V(x_t)` control：未记录。
- 第二个视觉编码器的完整matched结果：当前报告未整理。
- 另一模型规模的完整matched结果：当前报告未整理。
- 多seed均值与标准差：未记录。
- Feature Residual layer 4 standalone、Hidden Replacement layer 4/10 standalone等完整matched接口网格：不完整。Hidden Replacement layer 8 standalone已有53.94。
- 当前工作树中的`attnscaf-hidden-replacement-layer8-no-consistency-100k.yaml`已被设置为30K oracle checkpoint配置；53.94对应的100K运行应以checkpoint内保存的args或原始运行配置为准，避免误用当前同名YAML复现。
- Q/K/V/QKV完全matched消融：QKV结果来源仍需核实。
- 20K/30K/40K完整scaffold duration表：目前只有25K、30K及部分40K配置，结果表不完整。
- CFG结果未纳入当前主表。先前对话中提到2M时约1.46、在370K停止某辅助loss后约1.42，但对应精确配置与完整指标尚未写入本报告，使用前需重新核对。

## 8. 建议新窗口首先核实的信息

以下是数据管理问题，不是论文观点：

1. 新的AttnScaf block 4 + REPA block 8 smooth组合实验的100K结果及完整指标。
2. 30K oracle FID批量评估的实际入口、clean reference配对方式、class matching和VAE设置。
3. Hidden Replacement layer 8的53.94对应checkpoint args；当前同名YAML已经改为只训练到30K，不能直接代表该100K运行。
4. Smooth combined 400K的最终评估文件及所有指标。
5. 所有主表结果的VAE类型是否一致。
6. QKV FID 14.56对应的实际checkpoint args。
7. Vanilla SiT 39.20的完整评估协议和其余指标。
8. 每个表格行是否均为50K samples、CFG=1.0、SDE 250 steps、seed 0。

## 9. 新对话中应保持的事实边界

- 可以陈述：REPA将预训练视觉表征作为intermediate DiT feature的alignment target。
- 可以陈述：直接用投影后的clean-image representation完整替换中间hidden state，在当前block-8实验中失败；保持clean representation开启的oracle结果也很差。
- 可以陈述：K/V接口保留SiT native query和residual/noisy hidden pathway；当前oracle与标准撤除实验均明显优于完整hidden replacement。
- 可以陈述：在block 8，AttnScaf-only、+consistency、+smooth形成`22.45 → 19.08 → 17.22`的100K FID阶梯。
- 可以陈述：在block 4，standalone smooth结果为18.06；历史combined smooth + REPA + SpatialNorm结果为11.30，但该历史组合的REPA depth需要与新block-8组合明确区分。
- 不应陈述：完整hidden replacement相对Vanilla SiT带来小幅提升。
- 不应把3.91 oracle FID描述为标准生成性能或与REPA 19.40作公平比较。
- 在新的block-4 scaffold + block-8 REPA结果出来前，不应宣称分离两个位置一定优于同层设置，也不应预填最终组合数字。
- “同层监督重复”“早层天然更适合scaffold”等说法目前最多是待验证假设，需要matched depth ablation支持。
