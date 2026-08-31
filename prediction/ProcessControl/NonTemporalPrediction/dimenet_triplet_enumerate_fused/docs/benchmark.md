<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

## Baselines

- Stock PyG DimeNet++ is unavailable on node202: `radius_graph` requires
  `pyg-lib>=0.6.0`, and `triplets()` requires `torch-sparse`.
- The speed denominator is an equivalent resident NPU tensor path using
  `repeat_interleave`, `index_select`, masking, and fixed precomputed CSR.
- Graph-radius preprocessing is excluded from both paths. The compared model
  region begins at triplet construction and includes the otherwise identical
  DimeNet++ forward.
- Timings are synchronized no-grad wall-clock medians; H2D/D2H is excluded.

## Results

| Graphs | Nodes | Edges | Triplets | Resident stage | Custom stage | Stage speedup | Resident E2E | Custom E2E | Reduction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 259 | 3,972 | 59,038 | 19.285 ms | 2.244 ms | 8.59x | 105.355 ms | 87.964 ms | 16.51% |
| 32 | 465 | 6,452 | 87,590 | 28.151 ms | 3.102 ms | 9.07x | 154.420 ms | 129.704 ms | 16.01% |
| 64 | 944 | 13,042 | 173,204 | 54.998 ms | 5.576 ms | 9.86x | 308.468 ms | 247.693 ms | 19.70% |

Topology mismatch count and model maximum absolute error are zero in every
case. The fixed initialized model proves exact DimeNet++ semantics and E2E
latency only; there is no checkpoint-quality claim.

The broader runtime screen used regular degree-4/8/16 graphs. Custom latency
was 0.532/1.313/14.666 ms versus 2.132/11.444/154.885 ms for resident NPU.
