"""
TTM: Tiny Time Mixers (IBM Research)
=====================================

- Paper: ICML 2024
- Authors: Ekambaram et al., IBM Research
- Links: https://github.com/ibm-granite/granite-tsfm

核心思想：
  超轻量级时序基础模型 —— TSMixer 架构 + 多分辨率预训练。
  参数仅 ~1M，可直接部署在边缘设备和 DCS 仪表上。

关键特点：
  - 零样本/少样本预测
  - 支持微调适配特定设备/工况
  - 推理延迟极低，适合实时控制场景

化工场景价值：边缘 DCS 仪表实时预测、就地推理无需上云
"""