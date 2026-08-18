<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Benchmark

Ascend 910B3, CANN 8.1.RC1.alpha001, PyTorch 2.5.0, and torch_npu 2.5.1. Each
entry is the mean of two runs with two warmups and seven synchronized samples.

| Scale | `(B,N,K)` | NPU stage (ms) | Custom stage (ms) | Stage reduction | Subgraph E2E (ms) | Custom E2E (ms) | E2E reduction | Error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Small | `(4,512,64)` | 12.613 | 0.585 | 95.36% | 12.988 | 0.681 | 94.74% | `0` |
| Medium | `(16,1024,128)` | 25.133 | 1.934 | 92.30% | 24.229 | 2.058 | 91.50% | `0` |
| Large | `(40,2048,256)` | 48.229 | 7.272 | 84.92% | 48.715 | 7.440 | 84.73% | `0` |

The baseline is an equivalent iterative resident Torch NPU implementation. E2E
adds a representative gather and aggregation. These are inference-subgraph,
not whole-model, latencies; the CPU reference is for correctness only.

## Pretrained PointNet++ Model E2E

The operator was inserted into both FPS locations of the original PointNet++
SSG classification network: `(N=1024,K=512)` and `(N=512,K=128)`. The model
uses the repository's pretrained ModelNet40 checkpoint and real test samples
`airplane_0627` and `bathtub_0107`, cycled for batches above two.

| Batch | Baseline FPS | Custom FPS | FPS share of baseline E2E | Baseline model E2E | Custom model E2E | Reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 141.277 ms | 8.226 ms | 75.82% | 186.341 ms | 50.261 ms | 73.03% |
| 4 | 113.379 ms | 8.214 ms | 63.57% | 178.352 ms | 70.015 ms | 60.74% |
| 8 | 112.745 ms | 8.212 ms | 55.24% | 204.095 ms | 94.873 ms | 53.52% |

Both isolated and in-model FPS indices are identical, and complete model output
error is zero. The custom integration explicitly materializes the non-contiguous
PointNet++ `[B,N,3]` view before calling ACLNN; that copy is included in model
E2E. Results average two independent trials with three warmups and ten
synchronized samples per trial; the second trial reverses measurement order.

Reproduction driver and raw samples:

- `tests/model_e2e/benchmark_pointnet2_fps_e2e.py`
- `tests/model_e2e/results/pointnet2_fps_e2e_20260731.json`

After building `farthest_point_sampling_fused`, run the following command from
its package directory:

```bash
ASCEND_RT_VISIBLE_DEVICES=6 python3 tests/model_e2e/benchmark_pointnet2_fps_e2e.py \
  --pointnet2-source /path/to/pointnet2/models \
  --source-commit eb64fe0b4c24055559cea26299cb485dcb43d8dd \
  --checkpoint /path/to/best_model.pth \
  --point-clouds /path/to/airplane_0627_coord.npy /path/to/bathtub_0107_coord.npy \
  --labels 0 1 \
  --operator-build-dir build \
  --output tests/model_e2e/results/pointnet2_fps_e2e.json \
  --batches 1 4 8 --warmup 3 --repeat 10 --trials 2
```
