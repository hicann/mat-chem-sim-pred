# 表格类数据预测预训练模型（Tabular Data Pretrained Models）

## 领域简介

表格类数据（Tabular Data）是化工行业中最普遍的数据形态。从配方表、物性表、操作条件表到实验记录表，化工领域的结构化数据几乎总是以表格形式存在。特征类型包括数值列（温度、压力、浓度、配比）和类别列（催化剂种类、溶剂类型、装置编号）。

传统方法将梯度提升树（XGBoost / LightGBM / CatBoost）作为表格学习的基准。然而，随着深度学习和预训练技术在其他领域（NLP、CV）取得巨大成功，近年来越来越多研究尝试将预训练-微调范式引入表格数据。本方向的目标是在化工表格数据上构建通用预训练模型，通过大规模化工表格数据的自监督学习获得可迁移的表征，在下游任务（分类、回归、异常检测）上以少量微调样本取得高精度。

## 关键模型与算法

### TabTransformer

Google 提出的将 Transformer 应用于表格数据的混合架构：

- **类别特征**：通过可学习的 Embedding 层映射为连续向量
- **数值特征**：保持原始数值或 Layer Normalization 后直接拼接
- **混合编码器**：Category Embedding + Numerical Feature → Transformer Encoder → MLP Head
- **代表工作**：Huang et al., *arXiv* 2020。
- **算子化要点**：类别 Embedding 矩阵查找、Transformer 多头注意力前向传播。

### FT-Transformer

纯 Transformer 架构的表格模型，所有特征统一嵌入：

- **Feature Tokenizer**：将每个特征（不论类别或数值）编码为一个 Token（Embedding + 位置编码）
- **Transformer Encoder**：堆叠的 MHSA + FFN 层，捕获跨特征依赖
- **CLS Token / Pooling**：输出预测结果
- **代表工作**：Gorishniy et al., *NeurIPS* 2021。
- **算子化要点**：全特征 Tokenizer 与标准 Transformer 推理集成。

### TabNet

Google 提出的自带可解释性的表格神经网络：

- **Sparse Attention Mask**：在每个决策步选择重要特征，实现特征选择与主线学习的同时进行
- **Transformer-Style Encoder**：用 GLU（门控线性单元）和 Attention 机制
- **可解释性**：Attention Mask 的累加可直接解释每个特征的贡献度
- **代表工作**：Arik & Pfister, *Nat. Intell.* 2021。
- **算子化要点**：稀疏注意力掩码计算 + GLU 前向推理。

### Gradient Boosting 替代模型

用神经网络近似树模型的决策集成：

- **DeepGBM**：用两个神经网络（CatNN 处理类别特征、GBDT 处理数值特征）联合建模。Ke et al., *KDD* 2019。
- **NODE（Neural Oblivious Decision Ensemble）**：可微的决策树层，用 soft 分割替代硬分割，支持端到端梯度传播。Popov et al., *NeurIPS* 2019。

### 表格预训练方法

大规模化工表格上的预训练范式：

- **掩码列预测**：类似 BERT 的 Masked Feature Prediction。随机掩码部分列，预测被掩码的值。
- **对比学习**：SimCLR 风格的正负样本对构造（行采样 + 列扰动），在表征空间拉近同一样本的不同增广。
- **跨表迁移**：在不同表格间学练通用特征编码器，支持零样本迁移到新表。

### 缺失值生成

- 传统方法：均值填充、KNN 填充、MICE 多重插补
- 深度方法：基于扩散模型或 VAE 的缺失值生成，不局限于插值而是生成分布一致的合理样本

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 催化剂配方优化 | FT-Transformer / TabNet | 活性组分比例/载体/助剂 → 催化活性/选择性预测 |
| 聚合物性能预测 | TabTransformer | 单体配比/聚合条件 → 分子量/分布/转化率 |
| 工艺条件反向设计 | 表格预训练 + 贝叶斯优化 | 大量历史配方数据预训练 + 少量新目标微调 |
| 产品质量预测 | DeepGBM / NODE | 当前操作条件 → 产品关键质量指标 |
| 实验数据缺失填补 | 扩散缺失值生成 | 补齐因仪器/成本导致的缺失测量值 |

## 相关资源

- **UCI Tabular Benchmarks**：各类表格数据标准基准
- **TabZilla**（https://github.com/naszilla/tabzilla）：表格深度学习基准框架
- **CCFD（Chinese Chemical & Fertilizer Dataset）**：化工过程数据

## 本仓库中的定位

本方向规划中的算子/模型包括：

- TabTransformer 推理算子（类别 Embedding + 混合编码器）
- FT-Transformer 推理算子（Feature Tokenizer + 全 Transformer）
- TabNet 推理算子（Sparse Attention Mask + GLU）
- Gradient Boosting 替代算子（DeepGBM / NODE）
- 表格数据预训练算子（Mask 列预测 / 对比学习）
- 缺失值生成算子（扩散模型 / VAE）

> 详细规划请参见 [roadmap.md](../../roadmap.md)。