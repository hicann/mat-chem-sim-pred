"""
Timer: Generative Pre-trained Transformers for Time Series (THUML)
===================================================================

- Paper: ICML 2024 → Timer-S1 (2025, 83亿参数)
- Authors: Yong Liu et al., 清华大学 THUML / 字节跳动
- Links: https://github.com/thuml/Timer

核心思想：
  GPT 风格的 Decoder-only 生成式时序模型。将预测/插补/异常检测统一为生成任务。
  1 万亿时间点预训练 (TimeBench 数据集)，GIFT-Eval 基准 SOTA。

版本演进：
  Timer 1.0 (2024):  首次验证时序 Scaling Law
  Timer 2.0 (2024):  二维注意力（时间+变量维度）
  Timer 3.0 (2025):  生成式建模, 万亿点预训练, 推理 Chronos 的 20x
  Timer-S1 (2025):   8.3B 参数, TimeMoE+TimeSTP 串行预测, GIFT-Eval SOTA

化工场景价值：DCS 过程监控、反应器趋势预测、批次过程模式匹配
"""