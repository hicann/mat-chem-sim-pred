"""
TabPFN: Tabular Prior-data Fitted Network
=========================================

- Paper: ICLR 2023 (Oral) → Nature 2025 (v2)
- Authors: Noah Hollmann, Samuel Müller, Frank Hutter (Uni Freiburg)
- Links: https://github.com/PriorLabs/TabPFN

核心思想：
  在 1 亿+ 合成表格数据集上预训练 Transformer，通过 In-Context Learning (ICL)
  在单次前向传播中完成预测，无需梯度更新、无需超参数调优。

版本演进：
  v1 (ICLR 2023):  ≤1K 样本, ≤100 特征, 仅分类
  v2 (Nature 2025): ≤10K 样本, ≤500 特征, 分类+回归
  v2.5 (2025):     ≤100K 样本, 预训练权重公开
  Real-TabPFN (2025): 在 OpenML/Kaggle 真实数据上继续预训练，性能进一步提升

化工场景价值：小样本配方筛选、催化剂活性预测、实验数据建模
"""