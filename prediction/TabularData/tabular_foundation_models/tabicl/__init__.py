"""
TabICL: Tabular Foundation Model for In-Context Learning on Large Data
======================================================================

- Paper: ICML 2025
- Authors: Jingang Qu, David Holzmüller, Gaël Varoquaux, Marine Le Morvan (INRIA)
- Links: https://github.com/soda-inria/tabicl

核心思想：
  两阶段架构——先列后行注意力构建固定维行嵌入，再 Transformer 做高效 ICL。
  将 TabPFN 的 O(n²m+nm²) 复杂度降至 O(n²+nm²)。

关键能力：
  支持最多 500K 样本的表格分类，比 TabPFNv2 快 10 倍
  TabICLv2 (2025): QASSMax 注意力 + Muon 优化器，TabArena 基准上 SOTA

化工场景价值：大规模化工过程数据建模、生产批次质量预测
"""