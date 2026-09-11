# Benchmark

Environment: Ascend910B3, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, torch_geometric 2.9.0, FP32. Results are synchronized
resident-device wall-clock medians of 50 iterations after 10 warmups; H2D
and D2H are excluded. TorchAir was unavailable in the isolated environment.

The model is official PyG `SAGEConv(1433,32,aggr=mean) -> ReLU -> Linear(32,7)`
trained on all 10,556 directed Cora edges. The strongest baseline is the faster
of the official PyG call and an algebraically equivalent resident torch_npu
`index_add_` rewrite.

| Copies | Nodes | Stage baseline -> custom ms | Stage speedup | Model baseline -> custom ms | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,708 | 0.349 -> 0.217 | 1.613x | 0.535 -> 0.427 | 20.21% |
| 4 | 10,832 | 0.959 -> 0.444 | 2.162x | 1.191 -> 0.648 | 45.58% |
| 16 | 43,328 | 3.478 -> 1.344 | 2.589x | 4.548 -> 2.375 | 47.78% |
| 32 | 86,656 | 6.995 -> 2.588 | 2.702x | 8.975 -> 4.524 | 49.59% |
| 64 | 173,312 | 13.783 -> 5.034 | 2.738x | 17.719 -> 8.931 | 49.60% |

The replacement stage is `65.33%-80.58%` of the strongest transformed model
path. Baseline and custom test accuracy are both `0.7120`; prediction agreement
is approximately 100%, and maximum model error is `4.77e-7`. Raw evidence:
`tests/model_e2e/cora_pyg_sageconv_formal_20260801.json`.

The 64/256/1024/2708-node dispatch scan shows `1.81x-2.57x` component speedup
and maximum error `2.38e-7`; see `tests/model_e2e/csr_sage_dispatch_20260801.json`.
