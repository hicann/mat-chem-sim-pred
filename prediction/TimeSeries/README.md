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

### 一、Transformer 路线——古典变体

#### Informer

首个专门针对长序列时序预测的高效 Transformer：

- **ProbSparse 自注意力**：只选取 Query 中稀疏的 Top-u 主导注意力分数，降低计算复杂度
- **自注意力蒸馏**：跨层 MaxPooling/1D Conv 逐步压缩主导注意力特征
- **生成式 Decoder**：一次性输出全部预测步长，避免累积误差
- **代表工作**：Zhou et al., *AAAI* 2021 (Best Paper)。

#### Autoformer

自相关 Transformer——发现时序中的周期性依赖：

- **Auto-Correlation 机制**：基于 FFT 的快速自相关计算替代标准自注意力
- **Time Delay Aggregation**：选定 Top-k 自相关峰对应的时延，Roll 操作循环移位加权聚合
- **渐进式序列分解**：滑动平均将序列分解为趋势项和季节项
- **代表工作**：Wu et al., *NeurIPS* 2021。

#### FEDformer

频域增强分解 Transformer——注意力完全迁移到频域：

- **Fourier 增强模块**：在频域用少量随机选择的 Fourier 分量做注意力
- **Wavelet 增强模块**：用小波变换（DWT）多尺度分解信号
- 继承 Autoformer 的季节-趋势分解架构
- **代表工作**：Zhou et al., *ICML* 2022。

#### Crossformer

专为多变量时序预测中**变量间依赖建模**的 Transformer：

- **Dimension-Segment-Wise (DSW) 嵌入**：每个变量单独分 Patch 嵌入，保留变量身份
- **Two-Stage Attention (TSA)**：Cross-Time Attention + Cross-Dimension Attention
- Router 机制降低跨维度注意力复杂度
- **代表工作**：Zhang & Yan, *ICLR* 2023 Spotlight。

#### ETSformer

指数平滑 Transformer——经典统计方法与 Transformer 融合：

- **三路分解**：可学习的指数平滑核将序列分解为 Level、Trend、Seasonality 三个分量
- 各分量独立经过 Transformer encoder 处理，最终线性加权融合
- **代表工作**：Woo et al., *ICML* 2023。

### 二、Transformer 路线——2023-2024 前沿变体

#### PatchTST

片段级 Transformer——将时间序列分割为 Patch：

- **Patching 大幅降维**：将 seq_len=L 分割为 stride=S、patch_len=P 的 N 个 Patch
- **Channel-Independent**：每个变量独立建模，避免变量间噪声干扰
- **自监督预训练**：随机 Mask 部分 Patch 并重建（MAE 风格）
- **代表工作**：Nie et al., *ICLR* 2023。

#### iTransformer

倒置 Transformer——以变量为 Token：

- 每个变量的整条时间序列作为一个 Token，Transformer 在变量间做注意力
- FFN 内部的 MLP 捕获每个变量的时间依赖
- 变量数 V << 序列长度 L 时，计算量从 O(L²) 降至 O(V²)
- **代表工作**：Liu et al., *ICLR* 2024。

#### TimesNet

将 1D 时间序列转换为 2D 张量的创新视角：

- **FFT 频谱分析**：利用 FFT 从时序中提取显著周期
- **时序→图像**：将 1D 序列按周期倍数折叠为 2D 张量
- **2D 卷积**：在 2D 张量上使用 Inception 模块卷积，捕获周期内与周期间依赖
- **代表工作**：Wu et al., *ICLR* 2023。
- **当前状态**：已实现 PyTorch 参考代码 [timesnet.py](timesnet.py)。

### 三、卷积 / 线性 / MLP 高效路线

#### DLinear

极简线性分解模型——"简单却有效"的革命性发现：

- 滑动平均将序列分解为趋势分量和残差（季节）分量
- 各自仅用一个线性层（Wx + b）做预测
- 在多基准上超越 Inforer、Autoformer、FEDformer 等复杂 Transformer
- **代表工作**：Zeng et al., *AAAI* 2023。

#### TiDE

Google 的纯 MLP 时序预测模型——挑战"Transformer 是最佳选择"的假设：

- Encoder-Decoder 架构，全部为线性层 + 激活函数
- 支持外生变量（协变量）通过投影后与主表示拼接
- **代表工作**：Das et al., *TMLR* 2023。

#### ModernTCN

现代时序卷积网络——"时序版 ConvNeXt"：

- DWConv（Depthwise Separable Convolution）+ 倒瓶颈结构
- 大 kernel size（如 51, 101）扩大感受野
- 分通道独立建模 + 跨通道交互交替
- **代表工作**：Luo et al., *ICLR* 2024。

#### SCINet

样本卷积交互网络——基于二叉下采样树的层次化 CNN：

- SCI-Block：奇偶子序列分离 → 1D 卷积 → 交互融合
- 二叉树下采样：每过一层序列长度减半，形成 log(L) 层二叉树
- **代表工作**：Liu et al., *NeurIPS* 2022。

#### FreTS

频域 MLP——在频域用 MLP 做时序预测：

- RFFT 变换到频域 → 频域分量 MLP 全局映射 → IRFFT 变换回时域
- 频域分量具有全局性，少数主频分量已携带主要信息
- **代表工作**：Yi et al., *NeurIPS* 2023。

### 四、状态空间模型（SSM）路线

#### Mamba for TS

基于选择性 SSM 的长序列模型：

- **选择性 SSM 扫描**：根据输入内容动态选择状态转移矩阵，突破传统 SSM 线性时不变局限
- **线性复杂度**：序列长度 L 时复杂度 O(L)，远优于 Transformer 的 O(L²)
- 双向 Mamba（Bi-Mamba）捕获上下文，适合数万步超长 DCS 序列
- **代表工作**：Gu & Dao, *arXiv* 2023 (Mamba); 各时序适配变体发表于 *arXiv* 2024。

#### Koopa

基于 Koopman 算子理论——非线性时序动力学的线性化：

- Encoder 将时序嵌入到 Koopman 不变子空间
- 在线性空间中用可学习的 Koopman 算子矩阵做演化
- Decoder 将演化后的状态解码回时序，多层堆叠捕获多尺度动力学
- **代表工作**：Liu et al., *NeurIPS* 2023。

### 五、时序基础模型 / 预训练模型

- **TimesFM**：Google 的 Decoder-only Transformer 时序基础模型（~200M 参数），在数百万条跨领域时序上预训练，支持零样本预测。Das et al., *ICML* 2024。
- **Lag-Llama**：基于 LLaMA 架构的时序模型，Lag Features + 多频率支持 + 概率预测。Rasul et al., *arXiv* 2024。
- **Chronos**：Amazon 的 T5 Encoder-Decoder 时序基础模型，连续值量化 + 离散 Token 化。Ansari et al., *ICML* 2024。
- **Moirai**：Salesforce 的统一时序模型——跨频率、跨变量维度、跨预测长度的 Any-Variate Attention。Woo et al., *ICML* 2024。
- **TTM**：IBM 的超轻量时序基础模型（~1M 参数），TSMixer 架构 + 多分辨率预训练，可部署在边缘设备。Ekambaram et al., *ICML* 2024。

### 六、异常检测

化工过程监控中的时序异常检测：

- **Autoencoder**：重建误差作为异常得分，序列中重建误差大的窗口判定为异常
- **VAE**：概率重构 + KL 散度作为额外的异常信号
- **Anomaly Transformer**：关联差异机制（association discrepancy），多变量传感器通道同时监控
- **TS2Vec**：时间序列层级对比学习，通用的时序 Embedding 用于分类/回归/异常检测

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 产品关键质量指标预测 | PatchTST / iTransformer | 由当前过程变量序列预测未来时刻的产品质量 |
| DCS 传感器异常预警 | Anomaly Transformer / Autoencoder | 检测传感器漂移、堵塞、故障前的异常模式 |
| 批次过程状态识别 | TS2Vec + 分类头 | 对批次轨迹进行编码，识别不同配方/工况模式 |
| 长周期反应趋势预测 | Mamba / Koopa | 数万步反应器中温度和压力序列的趋势预测 |
| 过程控制优化 | TimesNet / FreTS | 从历史 DCS 数据中提取周期模式，辅助 PID/MPC 参数整定 |
| 边缘实时推理 | TTM / DLinear / TiDE | 算力有限的 DCS 现场仪表和边缘网关部署 |

## 综合对比矩阵

| 模型 | 年份 | 会议 | 核心技术路线 | 复杂度 | Ascend 适配难度 | 化工优先级 |
|------|------|------|-------------|--------|:---------------:|:----------:|
| **iTransformer** | 2024 | ICLR | Inverted Attention (V-Token) | O(V²) | 较高 | ★★★★★ |
| **PatchTST** | 2023 | ICLR | Patching + Transformer | O((L/P)²) | 较高 | ★★★★★ |
| **TimesNet** | 2023 | ICLR | FFT + 2D Conv | O(L log L) | 中高 | ★★★★ |
| **DLinear** | 2023 | AAAI | Linear Decomposition | O(L) | 极高 | ★★★ |
| **TTM** | 2024 | ICML | TSMixer (MLP-Mixer) | O(L) | 极高 | ★★★★★ |
| **TiDE** | 2023 | TMLR | Pure MLP Enc-Dec | O(L) | 极高 | ★★★★ |
| **ModernTCN** | 2024 | ICLR | Modern DWConv | O(L) | 较高 | ★★★★ |
| **Koopa** | 2023 | NeurIPS | Koopman Operator | O(L) | 较高 | ★★★★★ |
| **Crossformer** | 2023 | ICLR | Cross-Dimension Attention | O(L²+V²) | 中低 | ★★★★★ |
| **Autoformer** | 2021 | NeurIPS | Auto-Correlation + Decomp | O(L log L) | 中高 | ★★★★ |
| **FreTS** | 2023 | NeurIPS | Frequency MLP | O(L log L) | 较高 | ★★★★ |
| **Mamba TS** | 2023 | arXiv | Selective SSM | O(L) | 中低 | ★★★★★ |
| **Informer** | 2021 | AAAI | ProbSparse Attention | O(L log L) | 中等 | ★★★ |
| **TimesFM** | 2024 | ICML | Decoder-only FM | O((L/P)²) | 中等 | ★★★ |
| **Moirai** | 2024 | ICML | Any-Variate Attention | O((L/P)²) | 中低 | ★★★★ |
| **Chronos** | 2024 | ICML | T5 + Tokenization | O((L/P)²) | 中等 | ★★★ |

## 相关资源

- **UCR Time Series Archive**：时序分类标准基准
- **Monash Time Series Forecasting Repository**：预测标准基准
- **GluonTS**（https://github.com/awslabs/gluonts）：时序建模工具包
- **tsai**（https://github.com/timeseriesAI/tsai）：深度学习时序库
- **Time-Series-Library (TSLib)**（https://github.com/thuml/Time-Series-Library）：15+ 模型标准实现集合
- **TimesFM**（https://github.com/google-research/timesfm）：Google 时序基础模型
- **Chronos**（https://github.com/amazon-science/chronos-forecasting）：Amazon 时序基础模型
- **Moirai**（https://github.com/SalesforceAIResearch/uni2ts）：Salesforce 统一时序模型

## 本仓库中的定位

本方向规划中的算子/模型包括：

- TimesNet 推理算子（FFT + 2D 折叠 + 2D Conv）✅ PyTorch 参考实现
- PatchTST 推理算子（片段嵌入 + Transformer + 掩码重建）
- iTransformer 推理算子（倒置变量 Token Attention）
- DLinear 推理算子（趋势-季节分解 + 线性层）
- ModernTCN 推理算子（DWConv + 倒瓶颈结构）
- Koopa 推理算子（Koopman 算子 + 线性演化）
- Mamba/SSM 推理算子（线性复杂度的长序列建模）
- 时序基础模型推理算子（TimesFM / Chronos / Moirai / TTM）
- 时序预训练算子（对比学习 / 掩码重建）
- 异常检测算子（Anomaly Transformer / Autoencoder / VAE）

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。