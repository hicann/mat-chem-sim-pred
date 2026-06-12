# 表格类数据预测预训练模型（Tabular Data Pretrained Models）

## 领域简介

表格类数据（Tabular Data）是化工行业中最普遍的数据形态。从配方表、物性表、操作条件表到实验记录表，化工领域的结构化数据几乎总是以表格形式存在。特征类型包括数值列（温度、压力、浓度、配比）和类别列（催化剂种类、溶剂类型、装置编号）。

传统方法将梯度提升树（XGBoost / LightGBM / CatBoost）作为表格学习的基准。然而，随着深度学习和预训练技术在其他领域（NLP、CV）取得巨大成功，近年来越来越多研究尝试将预训练-微调范式引入表格数据。本方向的目标是在化工表格数据上构建通用预训练模型，通过大规模化工表格数据的自监督学习获得可迁移的表征，在下游任务（分类、回归、异常检测）上以少量微调样本取得高精度。

## 关键模型与算法

### 一、深度学习表格模型

#### TabTransformer

Google 提出的将 Transformer 应用于表格数据的混合架构：

- **类别特征**：通过可学习的 Embedding 层映射为连续向量
- **数值特征**：保持原始数值或 Layer Normalization 后直接拼接
- **混合编码器**：Category Embedding + Numerical Feature → Transformer Encoder → MLP Head
- **代表工作**：Huang et al., *arXiv* 2020。

#### FT-Transformer

纯 Transformer 架构的表格模型——当前最强的纯 Transformer 表格模型：

- **Feature Tokenizer**：将每个特征（不论类别或数值）编码为一个 Token（Embedding + 列偏置 + 缺失值指示符）
- **Transformer Encoder**：堆叠的 MHSA + FFN 层，捕获跨特征依赖
- **[CLS] Token / Pooling**：输出预测结果
- **代表工作**：Gorishniy et al., *NeurIPS* 2021。

#### TabNet

Google 提出的自带可解释性的表格神经网络：

- **Sparse Attention Mask**：在每个决策步选择重要特征，实现特征选择与主线学习的同时进行
- **Transformer-Style Encoder**：用 GLU（门控线性单元）和 Attention 机制
- **可解释性**：Attention Mask 的累加可直接解释每个特征的贡献度
- **代表工作**：Arik & Pfister, *Nat. Mach. Intell.* 2021。
- **当前状态**：已实现 PyTorch 参考代码 [tabnet.py](tabnet.py)。

#### TabPFN

德国弗莱堡大学 AutoML 组提出的革命性方法——表格数据的 In-Context Learning：

- 在数百万个合成表格任务上预训练 Transformer，学会贝叶斯推理
- 推理时不需要任何梯度更新——给定训练集作为 context，直接前向输出预测分布
- 小数据集（<1000 样本）上显著超越所有传统方法，天然适合化工小样本场景
- **代表工作**：Hollmann et al., *ICLR* 2023 (Oral)。
- **开源实现**：https://github.com/automl/TabPFN

#### SAINT

二维注意力 Transformer——行间注意力 + 列间注意力交替堆叠：

- 列间注意力捕获同一行不同特征间的关系（类似 FT-Transformer）
- 行间注意力跨样本捕获相似样本的模式，适合半监督/对比预训练
- **代表工作**：Somepalli et al., *NeurIPS* 2021 Workshop。

#### TabR

基于检索增强的表格模型——深度学习版 k-NN：

- 为每个测试样本从训练集中检索 k 个最相似样本
- 将检索样本的标签和特征通过注意力机制融合到预测中
- 小型表格数据集上超越 FT-Transformer
- **代表工作**：Gorishniy et al., *NeurIPS* 2023。

#### ExcelFormer

微软提出的"表格如 Excel 表"理念的 Transformer 模型：

- 行和列的双向注意力机制
- 特征掩码自监督预训练（随机掩码部分特征值，预测被掩码值）
- 回归任务上表现突出，掩码预训练与化工场景天然契合
- **代表工作**：Chen et al., *arXiv* 2023。

#### ModernNCA

极简架构的代表——仅 MLP 即可与复杂 Transformer 模型抗衡：

- 标准 MLP backbone + 基于 Soft Nearest Neighbor 的度量学习
- 训练和推理极快，适合工业部署的低延迟要求
- **代表工作**：Rubachev et al., *arXiv* 2024。

#### DANet

基于维度注意力的表格模型——针对大规模宽表（数千列）优化：

- 对每个特征维度分别计算注意力权重，取代传统全局 Softmax 注意力
- 低秩近似和分组注意力降低计算复杂度，化工宽表场景天然匹配
- **代表工作**：Chen et al., *2022*。

### 二、Gradient Boosting 替代模型

用神经网络近似树模型的决策集成：

- **DeepGBM**：用两个神经网络（CatNN 处理类别特征、GBDT 处理数值特征）联合建模。Ke et al., *KDD* 2019。
- **NODE（Neural Oblivious Decision Ensemble）**：可微的决策树层，用 soft 分割替代硬分割，支持端到端梯度传播。Popov et al., *NeurIPS* 2019。
- **GrowNet**：梯度提升神经网络——每轮训练浅层 MLP 拟合当前残差，逐轮累加。Badirli et al., *KDD* 2020。

### 三、表格预训练与跨表迁移

- **Trompt**：清华大学提出的表格预训练方法——特征 Token 重排增强 + 对比预训练 + Prompt Tuning 快速适配。Chen et al., *arXiv* 2023。
- **TransTab**：斯坦福 + MIT 提出的跨表迁移学习框架——将不同表格统一表示为 (样本, 列, 值) 三元组，支持多表联合预训练。Wang & Sun, *NeurIPS* 2022。
- **XTab**：微软 + 芝加哥大学的跨表预训练框架——共享列嵌入空间 + 表特定投影头，强调联邦学习场景。Zhu et al., *ICML* 2023。
- **T2G-Former**：将表格数据显式建模为图结构——样本间相似图 + 图 Transformer 学习结构化表征。Yan et al., *AAAI* 2023。

### 四、表格预训练方法

大规模化工表格上的预训练范式：

- **掩码列预测**：类似 BERT 的 Masked Feature Prediction。随机掩码部分列，预测被掩码的值。
- **对比学习**：SimCLR 风格的正负样本对构造（行采样 + 列扰动），在表征空间拉近同一样本的不同增广。
- **跨表迁移**：在不同表格间学练通用特征编码器，支持零样本迁移到新表。

### 五、缺失值生成

- 传统方法：均值填充、KNN 填充、MICE 多重插补
- 深度方法：基于扩散模型或 VAE 的缺失值生成，不局限于插值而是生成分布一致的合理样本

### 六、AutoML / 集成框架

- **AutoGluon-Tabular**：AWS 的 AutoML 框架——多模型 Stacking（LightGBM + CatBoost + XGBoost + 随机森林 + KNN + 神经网络），自动特征工程和超参搜索。Erickson et al., *AutoML Conference* 2020。
- **CatBoost / XGBoost / LightGBM**：作为 CPU 上的精度对照基准和知识蒸馏教师模型。

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 催化剂配方优化 | FT-Transformer / TabNet | 活性组分比例/载体/助剂 → 催化活性/选择性预测 |
| 聚合物性能预测 | TabTransformer / ModernNCA | 单体配比/聚合条件 → 分子量/分布/转化率 |
| 工艺条件反向设计 | 表格预训练 + 贝叶斯优化 | 大量历史配方数据预训练 + 少量新目标微调 |
| 产品质量预测 | DeepGBM / NODE | 当前操作条件 → 产品关键质量指标 |
| 实验数据缺失填补 | 扩散缺失值生成 | 补齐因仪器/成本导致的缺失测量值 |
| 小样本配方筛选 | TabPFN / TabR | 仅几十条实验记录即做零样本预测 |
| 跨表知识迁移 | TransTab / XTab | 从公开材料数据库迁移到企业内部配方数据 |

## 综合对比矩阵

| 模型 | 技术路线 | 小样本适用 | 可解释性 | Ascend 迁移难度 | 推荐优先级 |
|------|----------|:----------:|:--------:|:---------------:|:----------:|
| **FT-Transformer** | 纯 Transformer | 中 | 低 | 中等 | **P0** |
| **ModernNCA** | 极简 MLP | 中 | 低 | 极低 | **P0** |
| **TabNet** | 顺序注意力 | 中 | 高 | 中高 | **P0**（已实现 PyTorch） |
| **TabPFN** | In-Context 推理 | 极高 | 低 | 低（推理） | **P1** |
| **TransTab** | 跨表 Transformer | 高 | 低 | 中等 | **P1** |
| **TabTransformer** | 混合 Transformer | 中 | 低 | 中低 | **P1** |
| **TabR** | 检索增强 | 高 | 中 | 中等 | **P2** |
| **ExcelFormer** | 掩码预训练 | 高 | 低 | 中等 | **P2** |
| **Trompt** | 表格预训练 | 高 | 低 | 中等 | **P2** |
| **NODE** | 可微决策树 | 中 | 高 | 高 | **P3** |
| **SAINT** | 二维注意力 | 中 | 低 | 中高 | **P3** |
| **CatBoost/XGBoost/LightGBM** | CPU 基线 | 高 | 中 | 不适用 | 基线对照 |

## 相关资源

- **UCI Tabular Benchmarks**：各类表格数据标准基准
- **TabZilla**（https://github.com/naszilla/tabzilla）：表格深度学习基准框架
- **CCFD（Chinese Chemical & Fertilizer Dataset）**：化工过程数据
- **TabPFN**（https://github.com/automl/TabPFN）：表格 Prior-Data Fitted Networks
- **AutoGluon**（https://github.com/autogluon/autogluon）：AutoML 框架

## 本仓库中的定位

本方向规划中的算子/模型包括：

- TabTransformer 推理算子（类别 Embedding + 混合编码器）
- FT-Transformer 推理算子（Feature Tokenizer + 全 Transformer）
- TabNet 推理算子（Sparse Attention Mask + GLU）✅ PyTorch 参考实现
- TabPFN 推理算子（In-Context 表格推理）
- ModernNCA 推理算子（极简 MLP 基线）
- TransTab / XTab 跨表预训练推理算子
- Gradient Boosting 替代算子（DeepGBM / NODE / GrowNet）
- 表格数据预训练算子（Mask 列预测 / 对比学习）
- 缺失值生成算子（扩散模型 / VAE）

> 详细规划与待迁移模型清单请参见 [roadmap.md](../../roadmap.md) 和 [todo-algo-model.md](../../todo-algo-model.md)。