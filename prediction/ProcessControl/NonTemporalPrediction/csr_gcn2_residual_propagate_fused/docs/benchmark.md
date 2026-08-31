<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark Evidence

The exact source model is maintained PyG `examples/gcn2_cora.py`: 64
`GCN2Conv` layers, hidden width 64, `alpha=0.1`, and `theta=0.5` on Cora. The
replaced source stage is in `torch_geometric/nn/conv/gcn2_conv.py` at revision
`003c3cd8a10520567ceaeda619f0315e30ec2f66`. The three cases use one, two, and
four disjoint copies of Cora's 2,708 nodes and 13,264 normalized edges.

Environment: Ascend 910B3 device 6, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, PyG 2.9.0, FP32. TorchAir was unavailable. The strongest
baseline caches normalization and uses resident NPU `Index`, weighted
`index_add_`, scaling, and addition. Both paths retain all 64 trained `addmm`
calls, 65 ReLUs, input/output Linear layers, and LogSoftmax.

| Copies | Full model kernel (us) | Residual stage kernel (us) | Hotspot |
| ---: | ---: | ---: | ---: |
| 1 | 22,700.406 | 20,705.789 | 91.21% |
| 2 | 39,609.347 | 37,560.790 | 94.83% |
| 4 | 74,493.932 | 72,110.362 | 96.80% |

| Copies | Native stage (ms) | Custom stage (ms) | Speedup | Max error |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.405 | 0.262 | 1.55x | 4.17e-7 |
| 2 | 0.649 | 0.362 | 1.79x | 4.77e-7 |
| 4 | 1.187 | 0.564 | 2.11x | 5.36e-7 |

| Copies | Baseline E2E (ms) | Custom E2E (ms) | Reduction | Error | Accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 22.460 | 13.109 | 41.63% | 1.19e-6 | 79.20% / 79.20% |
| 2 | 39.360 | 17.928 | 54.45% | 1.91e-6 | 79.20% / 79.20% |
| 4 | 73.973 | 30.334 | 58.99% | 1.91e-6 | 79.20% / 79.20% |

The 100-epoch checkpoint SHA-256 is
`d2e86df046b65ae0ee71b477731ca02652b2708f08677b80bbdf0bd0a5991b2a`.
Prediction agreement is 100% for one/two copies and 99.999994% for four.
Formal raw results are retained as
`tests/model_e2e/gcn2_cora_formal_20260802.json` and
`tests/model_e2e/gcn2_cora_hotspot_formal_20260802.json`.
