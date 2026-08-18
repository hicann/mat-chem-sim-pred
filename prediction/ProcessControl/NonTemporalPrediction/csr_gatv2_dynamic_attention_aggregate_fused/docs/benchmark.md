<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The source model is a two-layer maintained PyG `GATv2Conv` classifier on full
Cora, hidden width 8, eight first-layer heads, one output head, normalized
features, and no CSR edge truncation. The checkpoint SHA-256 is
`46b8c8019b47bb63011394b3d6ee2ac25b805e9421f6776e9bcc62f654056357` and Cora
test accuracy is 79.60%. Environment: Ascend 910B3 device 6, CANN
8.1.RC1.alpha001, PyTorch/torch_npu 2.5, PyG 2.9.0, FP32.

| Copies | Nodes / edges | Native stage (ms) | Custom (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,708 / 13,264 | 28.465 | 1.796 | 15.85x | 5.96e-8 |
| 4 | 10,832 / 53,056 | 65.892 | 5.185 | 12.71x | 5.96e-8 |
| 8 | 21,664 / 106,112 | 78.959 | 9.026 | 8.75x | 5.96e-8 |

| Copies | Native E2E (ms) | Custom E2E (ms) | Reduction | Prediction agreement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 25.507 | 2.210 | 91.34% | 100% |
| 4 | 60.667 | 6.098 | 89.95% | 99.999994% |
| 8 | 82.040 | 10.814 | 86.82% | 99.999994% |

Complete resident-model profiler attribution is 99.27%/99.81%/99.80% for the
three scales, dominated by full-graph `Index` gathers. Official scatter has CPU
fallback in this environment; each E2E table uses the faster correct official
or resident baseline. Raw JSON is checked in under `tests/model_e2e/`.
