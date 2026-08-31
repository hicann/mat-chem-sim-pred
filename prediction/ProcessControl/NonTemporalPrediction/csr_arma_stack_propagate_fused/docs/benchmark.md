<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The exact source model is maintained PyG `examples/arma.py`: two `ARMAConv`
layers with `K=3`, `T=2`, shared weights, hidden width 16, root dropout 0.25,
and Cora input normalization. The replaceable stage occurs four times. Tests
use one, two, and four disjoint Cora copies; each graph has 2,708 nodes and
10,556 normalized directed edges without added self-loops.

Environment: Ascend 910B3 device 6, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, PyG 2.9.0, FP32. TorchAir was unavailable. The native baseline
keeps tensors resident on NPU and uses Index, edge multiplication,
`index_add_`, additions, and ReLU. Both paths keep all learned matmuls and the
final stack mean unchanged.

The checkpoint was trained for 400 epochs with the same normalized ARMA graph
using an equivalent dense NPU matmul because native `ARMAConv` backward fails
in this environment inside `aclnnScatterAdd`. Loading its weights into the
official PyG module and the staged model gives zero output error and 100%
prediction agreement. Its SHA-256 is
`0471c93f09b84f860a438016b638479f8c3e2d8ccce25d6757f529583e3863b9`.

| Copies | Full model kernel (us) | Stage kernel (us) | Hotspot |
| ---: | ---: | ---: | ---: |
| 1 | 2,993.040 | 2,302.996 | 76.95% |
| 2 | 5,342.994 | 4,373.375 | 81.85% |
| 4 | 9,913.422 | 8,466.390 | 85.40% |

Profiler attribution includes `Index`, `Mul`, `ZerosLike`,
`InplaceIndexAdd`, and the two stage additions. All ReLU kernels are excluded
because the model has one additional ReLU outside the replaceable stage.

| Copies | Native stage (ms) | Custom stage (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.709 | 0.504 | 1.41x | 1.79e-7 |
| 2 | 1.207 | 0.827 | 1.46x | 2.38e-7 |
| 4 | 2.242 | 1.509 | 1.49x | 2.98e-7 |

| Copies | Baseline E2E (ms) | Custom E2E (ms) | Reduction | Error | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 3.137 | 2.039 | 35.01% | 7.63e-6 | 79.50% / 79.50% |
| 2 | 5.431 | 3.536 | 34.90% | 7.63e-6 | 79.50% / 79.50% |
| 4 | 10.049 | 6.537 | 34.95% | 1.53e-5 | 79.50% / 79.50% |

Prediction agreement is 100% for one/two copies and 99.999994% for four.
Formal raw results are retained as
`tests/model_e2e/arma_cora_formal_20260802.json` and
`tests/model_e2e/arma_cora_hotspot_formal_20260802.json`.
