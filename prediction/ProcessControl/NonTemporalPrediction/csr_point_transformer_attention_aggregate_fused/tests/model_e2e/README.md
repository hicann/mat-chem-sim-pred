<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E

The audited benchmark uses held-out ModelNet10 points, a real PyG
`PointTransformerConv`, 16-NN, global pooling, and a shared prediction head.
`model_e2e_results_formal.json` retains the synchronized sample timings,
strong-baseline choice, shape, and correctness record. The separate TorchAir
audit was performed on the same operator build; its generated OPP harness is
not part of this source MR.
