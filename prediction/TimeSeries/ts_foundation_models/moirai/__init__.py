"""
Moirai: Unified Time Series Forecasting (Salesforce AI Research)
=================================================================

- Paper: ICML 2024 → Moirai-2 / Moirai-MoE (2025)
- Authors: Woo et al., Salesforce AI Research
- Links: https://github.com/SalesforceAIResearch/uni2ts

核心思想：
  Any-Variate Attention —— 原生支持任意维度多变量输入。
  Masked Encoder 预训练，3600万条时序/270亿观测点数据 (LOTSA)。

版本演进：
  Moirai 1.0 (ICML 2024):  Masked Encoder, multi-patch, mixture distribution
  Moirai 2.0 (2025):       Decoder-only, single patch, quantile loss, 2x faster, 30x smaller
  Moirai-MoE (2025):       稀疏专家混合, 11M 激活参数超越 311M 全参数模型

化工场景价值：多传感器变量联合预测、跨设备/跨产线泛化
"""