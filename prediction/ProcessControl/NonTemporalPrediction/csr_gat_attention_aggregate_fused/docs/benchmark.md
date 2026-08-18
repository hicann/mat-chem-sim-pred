<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The source model is the maintained PyG `examples/gat.py` topology: two
`GATConv` layers, hidden width 8, eight first-layer heads, one output head, and
Cora with normalized input features. The exact CSR contains all 10,556 source
edges plus 2,708 self-loops: 13,264 edges total, maximum row size 169, and zero
truncated edges.

Environment: Ascend 910B3 device 6, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, PyG 2.9.0, FP32. TorchAir was unavailable. Component and E2E
baselines choose the faster correct path between official PyG scatter-segment
execution and an exact full-graph resident padded NPU implementation.

The 200-epoch checkpoint is
`gat_cora_200e.pt`, SHA-256
`348b687a265ae4f064e2d77f7ee98c06b4dc65191267d894e159321f602a6f3f`.
Baseline and custom inference both reach 79.40% Cora test accuracy.

On this torch_npu release, official PyG scatter-segment execution falls back
to CPU. The exact resident padded implementation is therefore measured as a
second baseline, and every performance table uses the faster of the two. This
prevents the custom operator from being compared only with a deliberately
weak NPU implementation. The profiler uses the resident NPU baseline because
CPU fallback has no complete NPU kernel trace.

| Copies | Resident model kernel (us) | Stage kernel (us) | Hotspot |
| ---: | ---: | ---: | ---: |
| 1 | 33,621.188 | 33,399.635 | 99.34% |
| 4 | 259,259.181 | 258,879.308 | 99.85% |
| 8 | 461,660.236 | 460,932.911 | 99.84% |

Profiler attribution includes only `Index`, `LeakyRelu`, `MaskedFill`, `Mul`,
`ReduceSum`, and `SoftmaxV2`. In particular, learned projection `MatMul`, layer
`Add`, `ELU`, final mean, and data movement kernels are excluded. `Index`
alone accounts for 97.38%, 99.01%, and 98.94% of full-model kernel time.

| Copies | Strongest native stage (ms) | Custom stage (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 8.577 | 1.625 | 5.28x | 3.73e-8 |
| 4 | 22.213 | 4.698 | 4.73x | 4.47e-8 |
| 8 | 28.641 | 8.281 | 3.46x | 3.73e-8 |

| Copies | Strongest baseline E2E (ms) | Custom E2E (ms) | Reduction | Error | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.366 | 2.009 | 86.92% | 4.77e-7 | 79.40% / 79.40% |
| 4 | 39.187 | 5.561 | 85.81% | 7.15e-7 | 79.40% / 79.40% |
| 8 | 52.698 | 9.753 | 81.49% | 7.15e-7 | 79.40% / 79.40% |

Prediction agreement is 100% for one/four copies and 99.999994% for eight.
Each latency is the mean of two reverse-order trials after three warmups, with
ten timed samples per trial. Formal raw results are retained as
`tests/model_e2e/gat_cora_formal_20260802.json` and
`tests/model_e2e/gat_cora_hotspot_formal_20260802.json`.
