"""
TabDPT: Tabular Discriminative Pre-trained Transformer
======================================================

- Paper: NeurIPS 2025
- Authors: Junwei Ma, Valentin Thomas et al. (Layer 6 AI / TD Bank)
- Links: https://github.com/layer6ai-labs/TabDPT

核心思想：
  检索增强 + 自监督学习在真实表格数据上预训练表格基础模型。
  证明真实数据预训练优于纯合成数据，且遵循 Scaling Law。

关键发现：
  同时使用真实数据和合成数据预训练最快收敛
  模型和数据的 Scaling Law → 更大模型+更多数据 → 更好性能

化工场景价值：从公开材料数据库迁移到企业配方数据
"""