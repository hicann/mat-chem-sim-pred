"""
ConTextTab: Semantics-Aware Tabular In-Context Learner
======================================================

- Paper: 2025
- Authors: Marco Spinaci, Marek Polewczyk et al. (SAP)
- Links: https://github.com/SAP-samples/contexttab

核心思想：
  将语义理解融入 TabPFN 风格的 ICL 框架，弥补合成数据训练缺乏语义的缺陷。
  使用不同模态专用 Embedding + 大规模真实表格训练。

关键能力：
  在语义丰富的 CARTE 基准上设立新标准
  同时利用列名语义和数值结构

化工场景价值：结合化工领域列名语义（分子量、沸点、官能团等）
"""