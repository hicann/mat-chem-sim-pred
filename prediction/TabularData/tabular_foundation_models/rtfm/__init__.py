"""
RTFM: Robust Tabular Foundation Models
======================================

- Paper: AAAI 2026
- Authors: Matthew Peroni, Franck Le, Vadim Sheinin (IBM Research)

核心思想：
  对抗训练框架——在预训练期间自适应调整合成数据生成器，
  针对模型困难的分布生成训练数据，提升鲁棒性。

关键成果：
  应用于 TabPFNv2 分类器，AUC 提升最高 6%
  仅需 <100K 额外合成数据集即可实现

化工场景价值：增强对化工数据中异常值和分布漂移的鲁棒性
"""