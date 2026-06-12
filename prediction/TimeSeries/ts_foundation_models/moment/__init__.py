"""
MOMENT: A Family of Open Time-series Foundation Models (CMU)
=============================================================

- Paper: ICML 2024
- Authors: Goswami et al., Carnegie Mellon University / Auton Lab
- Links: https://huggingface.co/AutonLab/MOMENT-1-large

核心思想：
  首个面向通用时序分析的开源大规模预训练模型家族。
  一套模型覆盖预测、分类、异常检测、缺失值填补四大任务。

关键特点：
  - Time Series Pile: 13 领域, 1300 万条时序, 12.3 亿时间戳
  - Patching + T5-style Encoder-Decoder 架构
  - 冻结预训练 backbone + 线性头微调，可训练参数仅 6.29M

化工场景价值：DCS 异常预警、传感器漂移检测、缺失数据填补
"""