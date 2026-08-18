<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The source model is maintained PyG `examples/lightgcn.py` topology with two
propagation layers and 64-dimensional embeddings. The real task uses the
official MovieLens 100K `u1.base/u1.test` split: 943 users, 1,682 items,
80,000 train interactions represented as 160,000 directed edges, and 40,000
positive/negative evaluation pairs. Environment: Ascend 910B3 device 6, CANN
8.1.RC1.alpha001, PyTorch/torch_npu 2.5, PyG 2.9.0, FP32.

Checkpoint SHA-256 is
`3cdf33278377131cb408894d915e0dc8ab77c3c84bce40564731cc3e7216aeee`;
Recall@20 is 0.226475. Latency uses ten synchronized samples after three
warmups and the faster correct official PyG or resident-NPU baseline.

| Copies | Nodes / edges / pairs | Native stage (ms) | Custom (ms) | Speedup | Error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,625 / 160,000 / 40,000 | 6.542 | 2.206 | 2.97x | 3.87e-7 |
| 2 | 5,250 / 320,000 / 80,000 | 12.985 | 3.883 | 3.34x | 2.38e-7 |
| 4 | 10,500 / 640,000 / 160,000 | 27.450 | 7.370 | 3.72x | 2.38e-7 |

| Copies | Native E2E (ms) | Custom E2E (ms) | Reduction | Score error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 7.112 | 2.773 | 61.00% | 1.43e-6 |
| 2 | 14.028 | 4.963 | 64.62% | 1.43e-6 |
| 4 | 29.491 | 9.403 | 68.12% | 1.43e-6 |

Complete-model profiler attribution counts only propagation-exclusive
`InplaceIndexAdd` and `ZerosLike`, excluding shared Index/Mul kernels also used
by pair scoring. This conservative lower bound is 61.76%/62.14%/64.71%.
