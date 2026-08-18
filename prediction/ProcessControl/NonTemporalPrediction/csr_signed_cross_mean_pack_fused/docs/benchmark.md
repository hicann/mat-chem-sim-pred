<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

Environment: Ascend 910B3, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, PyG 2.9.0. Timings are synchronized NPU wall-clock medians
from 100 alternating A/B samples after 15 warmup iterations.

The complete model is maintained two-layer PyG SignedGCN trained on Bitcoin-OTC.
The real graph has 6,005 nodes, 25,624 positive training edges, 2,851 negative
training edges, and 7,117 test pairs. Larger cases are disconnected copies for
throughput scaling; the task split is unchanged. Checkpoint AUC is `0.8073` and
F1 is `0.9072`.

| Copies (nodes) | Component native -> custom ms | Speedup | Model native -> custom ms | E2E reduction | Hotspot |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 (6,005) | 1.4045 -> 0.3659 | 3.839x | 4.4244 -> 2.3061 | 47.88% | 50.83% |
| 4 (24,020) | 4.7992 -> 0.8787 | 5.462x | 9.0631 -> 4.3587 | 51.91% | 57.70% |
| 16 (96,080) | 18.6993 -> 2.9768 | 6.282x | 31.6421 -> 14.7981 | 53.23% | 60.75% |

An independent second 100-sample run measured model paths
`4.0246 -> 2.1132`, `9.0511 -> 4.3475`, and
`31.6449 -> 14.8056 ms`. The conservative reductions across both final-code
runs are `47.49% / 51.91% / 53.21%`; conservative component speedups are
`3.757x / 5.332x / 6.273x`.

The component max error is `4.47e-8` and the controlled second-layer embedding
error is `2.98e-8`. The unmodified official model contains nondeterministic NPU
scatter reductions: its self drift reaches `1.18e-2` in the retained final run.
Custom/official prediction agreement matches official self-agreement and rounds
to 100%; the controlled embedding prediction agreement is exactly 100%.

Level1 profiling uses a complete native replacement-equivalent model. Shape and
execution-order selection includes exactly four `index`, four `index_add_`, and
four degree `mul` calls from the second-layer pack. First-layer `scatter_add_`,
linear projections, discrimination head, allocation, and concatenation are
excluded, making the reported hotspot conservative.
