<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

Recorded FP32 run (build target `Ascend910B1`; runtime attribution corrected in
the consolidated evidence), Cora neighborhood hypergraph, `N=M=2708`, `E=13264`,
`H=2`, `C=8`, maximum segment 169, 20 synchronized no-grad samples:

| Path | Median |
| --- | ---: |
| PyG HypergraphConv attention eager | 4.890 ms |
| Equivalent resident eager | 2.879 ms |
| Custom model path | 1.314 ms |

The accepted result is 2.19x against resident eager, a 54.4% latency reduction.
TorchAir fullgraph fails at the missing `aten.scatter_reduce.two` converter.
The independent `C=24` clean component recheck is 1.52x with `1.86e-09` max error.
Raw accepted-shape samples are in
`../tests/model_e2e/model_e2e_hypergraph_c8_formal_run.json`.
