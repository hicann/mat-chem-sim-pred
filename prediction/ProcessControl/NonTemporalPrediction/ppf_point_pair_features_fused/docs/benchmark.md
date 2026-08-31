<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

The model benchmark uses real ModelNet10 point clouds in a three-layer PyG
PPFConv classifier. It compares the custom geometry operator with the fastest
correct resident NPU implementation while sharing the same aggregation and
same deterministic initialized weights.

| Clouds | Resident (ms) | Custom (ms) | Reduction | Max model error |
|---:|---:|---:|---:|---:|
| 4 | 68.470 | 2.941 | 95.70% | 2.98e-08 |
| 8 | 135.581 | 2.909 | 97.85% | 2.98e-08 |
| 16 | 264.694 | 3.915 | 98.52% | 4.47e-08 |

Prediction agreement is 1.0 at every scale. The source artifact is
`tests/model_e2e/ppfnet_modelnet10_e2e_final_20260818.json`. This is not an
official-checkpoint accuracy claim. Native PyG reaches a CPU fallback in the
audited path, so the resident implementation is the stronger denominator.
