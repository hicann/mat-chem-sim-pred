# Benchmark

Environment: Ascend910B3, CANN 8.1.RC1.alpha001, PyTorch 2.5,
torch_npu 2.5.1, torch_geometric 2.9.0, FP32. Results are synchronized
resident-device medians of 50 iterations after 10 warmups; H2D and D2H are
excluded. TorchAir was unavailable.

The model is maintained PyG
`Linear(1433,32) -> ReLU -> Linear(32,7) -> APPNP(K=10,alpha=0.1)` on all
10,556 input and 13,264 normalized Cora edges. Each baseline is the faster of
the official PyG call and the equivalent resident torch_npu `index_add_` path.

| Copies | Nodes | Stage baseline -> custom ms | Stage speedup | Model baseline -> custom ms | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,708 | 2.822 -> 1.443 | 1.956x | 2.932 -> 1.595 | 45.59% |
| 4 | 10,832 | 6.486 -> 4.544 | 1.427x | 6.622 -> 4.744 | 28.36% |
| 16 | 43,328 | 20.923 -> 16.670 | 1.255x | 20.969 -> 17.184 | 18.05% |
| 32 | 86,656 | 41.892 -> 32.376 | 1.294x | 42.782 -> 33.735 | 21.15% |
| 64 | 173,312 | 81.789 -> 64.881 | 1.261x | 83.867 -> 67.022 | 20.09% |

The strongest baseline changes from the resident rewrite on one Cora graph to
official PyG at larger shapes. CPU official/rewrite error is `1.19e-6`; NPU
official/rewrite error is at most `2.86e-6`; custom model error is at most
`5.72e-6`. Baseline and custom test accuracy are both `0.8270`.

The 64/256/1024/2708-node direct dispatch scan shows `2.22x-11.03x` speedup
against resident torch_npu and maximum error `1.19e-7`. Raw evidence is stored
under `tests/model_e2e/`.
