<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

Environment: Ascend910B3, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, torch_geometric 2.9.0, FP32. Results are synchronized
resident-device wall-clock medians of 50 iterations after 10 warmups; H2D and
D2H are excluded. TorchAir was unavailable in the isolated environment.

The workload is a PyG DGCNN graph classifier with three `GCNConv(32)` layers,
96 concatenated node channels, `SortAggregation(k=30)`, and an MLP. It uses all
188 molecular graphs from MUTAG (3,371 original nodes, 7 input features, two
classes). A reproducible 150/38 split trains the classification head; checkpoint
SHA256 is `c0eaeb3b41512354fcb4a347dbdd225e052bf75e43ba52b3ae43300145f40f9f`
and held-out accuracy is `0.8947`.

The strongest correct baseline is the faster of the official PyG model call
and an algebraically equivalent resident torch_npu GCN path. The equivalent
path wins in all rows. Its node output differs from official PyG by at most
`1.79e-7`; it is also used for deterministic checkpoint regression.

| Graphs | Nodes | Pool baseline -> custom ms | Pool speedup | Model baseline -> custom ms | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 16 | 278 | 2.327 -> 0.159 | 14.64x | 3.091 -> 1.284 | 58.47% |
| 64 | 1,161 | 3.040 -> 0.192 | 15.87x | 3.940 -> 1.327 | 66.33% |
| 188 | 3,421 | 5.276 -> 0.276 | 19.14x | 6.448 -> 1.405 | 78.20% |
| 376 | 6,847 | 8.488 -> 0.418 | 20.29x | 10.113 -> 1.994 | 80.28% |

The replacement stage is `75.29%-83.93%` of the strongest full-model path.
All prediction agreements are 100%; baseline and custom accuracies are equal
for every row, and maximum model error is `2.87e-6`. Raw evidence is stored in
`tests/model_e2e/mutag_pyg_dgcnn_formal_20260801.json`.

A separate 4/16/64/188-graph synthetic dispatch scan reports
`12.79x-13.32x` component speedup and zero error; see
`tests/model_e2e/sort_aggregation_dispatch_20260801.json`.
