"""
TimesFM: Time Series Foundation Model (Google Research)
========================================================

- Paper: ICML 2024 → TimesFM 2.5 (2025)
- Authors: Google Research
- Links: https://github.com/google-research/timesfm

核心思想：
  Decoder-only Transformer，将时间序列切分为 Patch，自回归预测。
  在 1000+ 亿时序数据点上预训练，支持零样本预测。

版本演进：
  v1.0 (ICML 2024): 200M params, 2048 context
  v2.0 (2025):       500M params
  v2.5 (2025):       200M params (优化), 16K context, 概率预测 + 外部协变量

化工场景价值：过程变量预测、产品关键质量指标趋势预测
"""