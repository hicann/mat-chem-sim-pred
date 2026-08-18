<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

## Environment

- Device: Ascend 910B3, device 6
- CANN: 8.1.RC1.alpha001
- PyTorch: 2.5; torch_npu: 2.5.1; PyG: 2.9.0
- PyG revision: `003c3cd8a10520567ceaeda619f0315e30ec2f66`
- Dataset: PPI deterministic first two-graph training batch
- Base shape: 3,144 nodes, 62,022 edges, 50 features, 121 labels
- Model: maintained `examples/film.py`, four layers, hidden size 320
- Checkpoint: 50 epochs, seed 20260802, test micro-F1 0.8866
- Checkpoint SHA256: `641cc810fefb4c07d2c64220f1ab795479cecab6cbe4c93e1fe133afaadc1bca`
- Timing: 3 warmups, 10 measured runs, two trials; median of trial medians

## Results

Official PyG is faster than the exact static-degree resident implementation at
all three sizes and is therefore the E2E baseline.

| Graphs | Replaceable native | Custom | Stage speedup | Official model | Custom model | E2E reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2.554 ms | 0.611 ms | 4.18x | 9.034 ms | 3.588 ms | 60.29% |
| 4 | 6.023 ms | 1.035 ms | 5.82x | 17.055 ms | 5.278 ms | 69.05% |
| 8 | 12.121 ms | 1.824 ms | 6.65x | 32.952 ms | 9.123 ms | 72.31% |

Maximum component absolute error is `7.6294e-6`. Deep-model maximum absolute
logit errors are `0.0625/0.09375/0.125`, while the corresponding error divided
by maximum output magnitude is only `1.28e-7/1.91e-7/2.56e-7`. Binary
prediction agreement is 100% at all sizes.

## Profiler attribution

One isolated first-layer stage consumes 23.60%, 24.29%, and 23.72% of exact
resident full-model NPU kernel time. The model invokes the compatible stage
four times. At the base shape, the isolated native stage launches ten kernels;
`Index` and `InplaceIndexAdd` dominate.

Raw samples are in `tests/model_e2e/film_ppi_formal_20260802.json`; profiler
aggregation is in `tests/model_e2e/film_ppi_hotspot_formal_20260802.json`.
