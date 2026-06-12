"""
TabuLa-8B: LLM-based Tabular Foundation Model
==============================================

- Paper: NeurIPS 2024
- Authors: Josh Gardner, Juan C. Perdomo, Ludwig Schmidt (Stanford / UW)
- Links: https://github.com/penfever/Tabula-8B

核心思想：
  将 Llama 3-8B 在 300 万张序列化表格上微调，实现 Few-Shot 表格预测。
  利用 LLM 的语义理解能力 + 世界知识处理表格数据。

关键特点：
  将表格行序列化为自然语言文本
  利用 LLM 预训练中已编码的列名语义（如"年龄""收入"等）
  支持上下文最多 32-64 个样本

化工场景价值：利用列名语义（如"温度""压力""催化剂"）辅助建模
"""