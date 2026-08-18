<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The source model is a two-layer maintained PyG `TransformerConv` classifier on
full Cora, hidden width 8, eight first-layer heads, one output head, normalized
features, and no edge features. TransformerConv's native root skip remains in
the model. The checkpoint SHA-256 is
`2ff5d48727b066ba4be8bddc46705e3a5dc0be048c14f8bafc072af1a1f9397f`; Cora test
accuracy is 78.70%. Environment: Ascend 910B3 device 6, CANN 8.1.RC1.alpha001,
PyTorch/torch_npu 2.5, PyG 2.9.0, FP32.

| Copies | Nodes / edges | Native stage (ms) | Custom (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2,708 / 10,556 | 12.122 | 1.523 | 7.96x | 2.09e-7 |
| 4 | 10,832 / 42,224 | 22.461 | 4.403 | 5.10x | 2.09e-7 |
| 8 | 21,664 / 84,448 | 34.810 | 7.531 | 4.62x | 2.09e-7 |

| Copies | Native E2E (ms) | Custom E2E (ms) | Reduction | Prediction agreement |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16.159 | 2.078 | 87.14% | 100% |
| 4 | 31.804 | 5.523 | 82.63% | 99.999994% |
| 8 | 43.829 | 9.554 | 78.20% | 99.999994% |

Complete resident-model profiler attribution is 99.12%/99.48%/99.70% for the
three scales. Learned Q/K/V and root skip MatMul, ELU, and head mean are
excluded. Raw JSON is checked in under `tests/model_e2e/`.
