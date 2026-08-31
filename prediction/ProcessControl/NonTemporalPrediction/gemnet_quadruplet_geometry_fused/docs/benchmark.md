<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

The benchmark runs the official GemNet geometry call path on real QM9 batches
and compares it with the custom ACLNN operator using identical topology and
deterministic initialized weights.

| Graphs | Resident (ms) | Custom (ms) | Speedup | Angle max error | Model max error |
|---:|---:|---:|---:|---:|---:|
| 1 | 224.944 | 34.121 | 6.59x | 5.01e-06 | 7.63e-06 |
| 2 | 375.105 | 44.922 | 8.35x | 5.01e-06 | 7.63e-06 |

The independent artifact is
`tests/model_e2e/gemnet_geometry_e2e_independent_20260818.json`. The official
source required a small `torch_scatter` compatibility shim in this environment.
The result is not an official-checkpoint accuracy claim, and TorchAir was not
runnable for this source path.
