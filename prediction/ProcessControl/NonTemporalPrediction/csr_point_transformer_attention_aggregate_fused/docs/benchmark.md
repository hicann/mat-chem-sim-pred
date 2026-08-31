<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

Recorded FP32 run (build target `Ascend910B1`; runtime attribution corrected in
the consolidated evidence), ModelNet10 held-out cache, 8 clouds x 256 points, 16-NN,
`C=32`, 5 warmups and 20 synchronized samples:

| Path | Median |
| --- | ---: |
| PyG PointTransformerConv eager | 12.236 ms |
| Equivalent resident eager | 13.932 ms |
| Custom model path | 2.638 ms |

The no-grad custom path is 4.64x faster than the strongest correct baseline.
Native TorchAir still lacks the `aten.scatter_reduce.two` converter; the
packaged custom-op converter and OPP pass FP32 `fullgraph=True` with `2.38e-7`
maximum error. Timings include projections, positional encoding, pooling, and
prediction head.
