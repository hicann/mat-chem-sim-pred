# 小数据预测优化模型（Small Data Prediction & Optimization）

## 领域简介

化工行业面临一个普遍挑战：**标记数据稀缺**。原因包括：

- **实验成本高昂**：一次催化活性测试、力学实验或反应条件筛选需要数小时到数天
- **长周期测试**：稳定性测试、老化实验、甚至数月的批次反应
- **安全与法规限制**：涉及危险品或受监管物质的实验受限
- **数据私密性**：企业不愿共享核心工艺数据

因此，化工 AI 场景中常见的情况是只有几百甚至几十个标记样本。在此类"小数据"环境下的预测与优化，无法直接套用大数据深度学习范式，需要专门的方法：高斯过程回归、贝叶斯神经网络、主动学习、小样本学习、迁移学习等。这些方法的核心特点是：**显式建模不确定性、利用先验知识、高效利用有限样本**。

本方向聚焦化工行业的小数据场景，为配方优化、工艺参数搜索、新催化剂开发等提供从数据采集到模型部署的完整方案。

## 关键模型与算法

### 高斯过程回归（Gaussian Process Regression）

GP 是小数据回归的经典方法，源自统计学和机器学习：

- **核函数**：RBF（径向基函数）、Matern（控制平滑度）、周期核（化学周期性）、线性核（趋势）等
- **预测输出**：均值 $\mu(x)$ + 方差 $\sigma^2(x)$（不确定性量化）
- **训练**：通过最大化对数边际似然优化核超参数
- **推理**：$O(n^3)$ 的 Cholesky 分解（$n$ 为训练样本数），对大规模数据需要稀疏近似
- **算子化要点**：Cholesky 分解 / 共轭梯度求解（线性系统）、核矩阵构建（pairwise 距离 + 核函数计算）

### 贝叶斯神经网络（Bayesian Neural Network, BNN）

将贝叶斯推断引入神经网络权重的学习方法：

- **变分 Dropout**（MC Dropout）：在推理时开启 Dropout，多次前向采样近似后验分布。Gal & Ghahramani, *ICML* 2016。
- **Laplace 近似**：在 MAP 解附近用高斯分布近似后验，通过 Fisher 信息矩阵估计不确定性
- **SWAG**：Stochastic Weight Averaging Gaussian，收集 SGD 轨迹上的权重快照近似后验
- **预测输出**：预测均值 + 模型不确定性（epistemic）+ 数据不确定性（aleatoric）

### 主动学习（Active Learning）

在标记预算有限时，主动选择最有标注价值的数据：

- **不确定性采样**：熵（Entropy）、边际（Margin）、BALD（Bayesian Active Learning by Disagreement）
- **多样性采样**：CoreSet（在嵌入空间中选择覆盖整个候选池的子集）、TypClust（典型性聚类）
- **混合策略**：不确定性 × 多样性 的组合采样函数
- **化工场景**：实验设计（Design of Experiments）的 ML 增强版

### 贝叶斯优化（Bayesian Optimization, BO）

全局黑箱优化的标准方法，尤其适合实验/仿真代价高昂的情况：

- **代理模型**：GP / RF / BNN 作为代理模型，提供预测均值和不确定性
- **采集函数**：EI（Expected Improvement）、PI（Probability of Improvement）、UCB（Upper Confidence Bound）
- **多目标优化**：EHVI（Expected Hypervolume Improvement）、ParEGO
- **带约束优化**：约束采集函数（Expected Constrained Improvement）
- **算子化要点**：GP 推理（均值 + 方差）+ 采集函数计算 + 约束处理

### 小样本迁移学习（Few-shot Transfer Learning）

从源域（公开数据集+模拟数据）预训练到目标域（少量化工实验数据）微调：

- **标准 pipeline**：大规模预训练（如通用分子表征或 Materials Project 数据）→ 冻结特征提取器或全量微调
- **参数高效微调**：Adapter、LoRA、Prefix-tuning 在化工场景下的适配
- **域适应**：最大均值差异（MMD）或对抗训练缩小源域和目标域的分布差异

### 度量学习与原型网络（Metric Learning & Prototypical Networks）

小样本分类场景的标准方案：

- **Prototypical Network**：每个类的原型 = 该类支撑集样本嵌入的均值，查询点分类到最近原型的类别。Snell et al., *NeurIPS* 2017。
- **Siamese Network**：共享权重的双塔网络，学习两个样本的相似性
- **化工场景**：故障类型识别、催化剂快速分类、反应路径判型

### 数据增强（Data Augmentation）

从少量真实样本生成合理合成样本：

- **VAE 增强**：在潜在空间中采样新样本，解码回原始特征空间
- **GAN 增强**：条件 GAN（cGAN）以类别或属性为条件生成新样本
- **扩散模型增强**：最近兴起的数据增强方法，生成质量高、模式覆盖广
- **化工特定增强**：反应条件扰动（温度 ±5K、压力 ±0.1MPa）、配方配比插值

## 典型化工应用

| 应用场景 | 推荐方法 | 说明 |
|----------|----------|------|
| 催化剂配方的小样本优化 | 贝叶斯优化 + GP | 仅 20-50 次实验找到最优催化剂配比 |
| 聚合物合成条件搜索 | 主动学习 + BNN | 从初始少样出发逐步探索最优聚合条件 |
| 设备故障类型识别 | Prototypical Network | 每类故障仅 1-5 个样本的极端小样本分类 |
| 新反应路径筛选 | GP 分类器 + 迁移学习 | 从已知反应迁移先验知识，预测新反应的可行性 |
| 实验设计（DoE）增强 | 主动学习采样 | 在高维工艺参数空间中选择最有效的实验点 |

## 相关资源

- **GPyTorch**（https://github.com/cornellius-gp/gpytorch）：高斯过程高效工具库
- **BoTorch**（https://github.com/pytorch/botorch）：贝叶斯优化库
- **scikit-optimize**（https://github.com/scikit-optimize/scikit-optimize）：轻量 BO 工具
- **learn2learn**（https://github.com/learnables/learn2learn）：小样本学习框架
- **Open Catalyst Project**（https://opencatalystproject.org/）：催化领域公开基准

## 本仓库中的定位

本方向规划中的算子/模型包括：

- 高斯过程回归算子（核矩阵构建 + Cholesky 分解/共轭梯度）
- 贝叶斯神经网络算子（MC Dropout / Laplace 近似推理）
- 主动学习采样算子（不确定性 / 多样性采样）
- 贝叶斯优化算子（EI / PI / UCB 采集函数 + 多目标支持）
- 小样本迁移学习算子（预训练→微调 pipeline）
- 度量学习/原型网络算子（Prototypical / Siamese Network）
- 数据增强算子（VAE / GAN / Diffusion 生成）

> 详细规划请参见 [roadmap.md](../../roadmap.md)。