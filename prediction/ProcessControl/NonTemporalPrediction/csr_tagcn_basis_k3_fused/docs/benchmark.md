<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The exact source model is maintained PyG `examples/tagcn.py`: two
`TAGConv(K=3)` layers with hidden width 16 on Cora. The operator replaces the
three-hop loop in `torch_geometric/nn/conv/tag_conv.py` at revision
`003c3cd8a10520567ceaeda619f0315e30ec2f66`. Cora has 2,708 nodes, 1,433 input
features, seven classes, and 10,556 normalized edges. The three cases use one,
two, and four disjoint graph copies.

Environment: Ascend 910B3 device 6, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, PyG 2.9.0, FP32. TorchAir was unavailable. The strongest native
baseline caches identical normalization and uses resident torch_npu `Index`,
weighted multiply, `index_add_`, and stack. Both paths retain all eight trained
Linear calls.

| Copies | Full model kernel (us) | Basis kernel (us) | Basis hotspot |
| ---: | ---: | ---: | ---: |
| 1 | 2,265.393 | 1,987.300 | 87.72% |
| 2 | 4,658.267 | 4,230.776 | 90.82% |
| 4 | 9,708.725 | 8,893.622 | 91.60% |

| Copies | Native basis (ms) | Custom basis (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.327 | 0.615 | 2.16x | 5.96e-8 |
| 2 | 2.954 | 1.028 | 2.87x | 5.96e-8 |
| 4 | 6.417 | 1.763 | 3.64x | 5.96e-8 |

| Copies | Baseline E2E (ms) | Custom E2E (ms) | Reduction | Model error | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2.324 | 1.327 | 42.91% | 5.72e-6 | 79.60% / 79.60% |
| 2 | 4.700 | 1.994 | 57.57% | 3.81e-6 | 79.60% / 79.60% |
| 4 | 9.788 | 3.697 | 62.23% | 3.81e-6 | 79.60% / 79.60% |

The 100-epoch checkpoint SHA-256 is
`44e91556e55250b4516a5c0f12d587b4cd7a53aa7646775c865e931f8d6c01b0`.
Prediction agreement is 100% for one/two copies and 99.999994% for four. Formal
library raw JSON and reproduction scripts are under `tests/model_e2e`.
