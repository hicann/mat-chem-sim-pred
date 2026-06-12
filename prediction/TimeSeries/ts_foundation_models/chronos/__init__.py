"""
Chronos: Learning to Tokenize Time for Forecasting (Amazon AWS)
================================================================

- Paper: ICML 2024 → Chronos-2 (2025)
- Authors: Ansari et al., Amazon AWS AI Labs
- Links: https://github.com/amazon-science/chronos-forecasting

核心思想：
  将时间序列值量化成离散 Token（类似 NLP 分词），用 T5 Encoder-Decoder
  预测下一个 Token 的概率分布。将时序预测转化为语言建模问题。

版本演进：
  Chronos (ICML 2024):   单变量, 9M-710M params
  Chronos-Bolt (2025):   推理加速优化
  Chronos-2 (2025.10):   支持多变量 + 协变量, HuggingFace 百万下载

化工场景价值：DCS 多传感器协同预测、批次反应轨迹预测
"""