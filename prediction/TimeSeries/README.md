# 时序类数据预测预训练模型（Time Series Pretrained Models）

## 领域简介

化工过程数据在时间维度上具有天然的时序特征。DCS（Distributed Control System）持续记录的温度、压力、流量、液位等过程变量、批次反应轨迹、质量指标的在线测量——这些数据以多变量时间序列的形态存储，构成了化工行业最重要的数据资产之一。

时序建模在化工中的核心任务包括：
- **预测**：未来时间步的过程变量或质量指标
- **异常检测**：识别工艺偏离、设备故障、安全风险
- **分类**：批次过程的状态识别与模式匹配
- **表征**：将时序序列压缩为信息丰富的向量用于下游预测

近年来，时序领域的深度学习模型快速发展，从 RNN/LSTM 到 Transformer、CNN、状态空间模型形成了多条技术路线。跟随 NLP 领域的趋势，**时序预训练** 也得到了越来越多的关注，旨在从海量化工 DCS 数据中预训练通用的时序编码器，支持少量标注即可适配新场景。

## 关键模型与算法

### TimesNet

将 1D 时间序列转换为 2D 张量的创新视角：

- **FFT 频谱分析**：利用 FFT 从时序中提取显著周期
- **时序→图像**：将 1D 序列按周期倍数折叠为 2D 张量（多种周期对应多种折叠方式）
- **2D 卷积**：在 2D 张量上使用 Inception 模块卷积，捕获周期内（intra-period）与周期间（inter-period）依赖
- **代表工作**：Wu et al., *ICLR* 2023。
- **算子化要点**：FFT 周期提取 → 序列 reshape → 2D Conv 推理。

### PatchTST

将时间切分为子序列片段（Patch）的 Transformer 模型：

- **Patching**：将长度为 L 的时序切分为若干长度为 P 的重叠或非重叠 Patch
- **Transformer Encoder**：每个 Patch 作为 Token 输入，学习 Patch 间的长程依赖
- **预训练方式**：随机掩码部分 Patch 并重建（类似于 BERT 的 Masked Autoencoding）
- **长序列预测**：Patch 显著减少了 Token 数量，使 Transformer 可处理更长的历史窗口
- **代表工作**：Nie et al., *ICLR* 2023。
- **算子化要点**：Patch 分割 + Transformer 多头注意力 + 掩码重建。

### TimesFM

Google 提出的时序基础模型（Time Series Foundation Model）：

- **Decoder-Only 架构**：延续 LLM 的预训练成功经验，使用 Decoder-only Transformer
- **大规模预训练**：在谷歌规模的时序数据上预训练（涵盖不同频率、不同领域）
- **零样本迁移**：预训练后未经微调即可在未见过的时序预测任务上取得竞争力
- **代表工作**：Das et al., *arXiv* 2024。
- **算子化要点**：Decoder-only 推理 + 位置编码 + 前缀 Token。

### Mamba / 状态空间模型（SSM）

新一代长序列高效建模方法：

- **Mamba 架构**：基于状态空间模型（SSM）的选择性扫描机制，输入相关的时间动态参数化
- **线性复杂度**：对序列长度的计算复杂度为 $O(L)$，远低于 Transformer 的 $O(L^2)$
- **长程依赖**：无限有效感受野，适合超长的化工过程序列（数万时间步）
- **代表工作**：Gu & Dao, *arXiv* 2023（Mamba）。
- **算子化要点**：选择性 SSM 扫描 + 卷积/离散化参数计算。

### 时序预训练方法

面向化工 DCS 数据的自监督预训练：

- **对比学习**：TS2Vec（时间序列层级对比）、TNC（时间邻近对比）、CoMT（跨模态时序对比）
- **掩码重建**：PatchTST 风格的 Masked Autoencoding、TimesNet 风格的重建
- **预训练输出**：通用的时序 Embedding，可迁移到分类、回归、异常检测等下游

### 异常检测

化工过程监控中的时序异常检测：

- **Autoencoder**：重建误差作为异常得分，序列中重建误差大的窗口判定为异常
- **VAE**：概率重构 + KL 散度作为额外的异常信号
- **Transformer 异常检测**：Anomaly Transformer 的关联差异机制（association discrepancy）
- **多变量支持**：化工 DCS 的数百维传感器通道的同时监控

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 产品关键质量指标预测 | PatchTST / TimesFM | 由当前过程变量序列预测未来时刻的产品质量 |
| DCS 传感器异常预警 | Anomaly Transformer / Autoencoder | 检测传感器漂移、堵塞、故障前的异常模式 |
| 批次过程状态识别 | TS2Vec + 分类头 | 对批次轨迹进行编码，识别不同配方/工况模式 |
| 长周期反应趋势预测 | Mamba / SSM | 数万步反应器中温度和压力序列的趋势预测 |
| 过程控制优化 | TimesNet | 从历史 DCS 数据中提取周期模式，辅助 PID/MPC 参数整定 |

## 相关资源

- **UCR Time Series Archive**：时序分类标准基准
- **Monash Time Series Forecasting Repository**：预测标准基准
- **GluonTS**（https://github.com/awslabs/gluonts）：时序建模工具包
- **tsai**（https://github.com/timeseriesAI/tsai）：深度学习时序库

## 本仓库中的定位

本方向规划中的算子/模型包括：

- TimesNet 推理算子（FFT + 2D 折叠 + 2D Conv）
- PatchTST 推理算子（片段嵌入 + Transformer + 掩码重建）
- TimesFM 推理算子（Decoder-only 时序基础模型）
- Mamba/SSM 推理算子（线性复杂度的长序列建模）
- 时序预训练算子（对比学习 / 掩码重建）
- 异常检测算子（Autoencoder / VAE / Transformer）

> 详细规划请参见 [roadmap.md](../../roadmap.md)。