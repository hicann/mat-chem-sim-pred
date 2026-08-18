<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Benchmark

Environment: node202, Ascend 910B3 physical device 6, CANN
8.1.RC1.alpha001, PyTorch 2.5.0, torch_npu 2.5.1. Each result averages two
trials; every trial uses 3 warmups and 10 synchronized samples.

Model: pretrained PointNet++ SSG classifier, checkpoint SHA256
`736a7136104fee7e92de99516bfd915c8b66b1c5432955dc37aa07f5808f6635`,
with real normalized ModelNet40 point clouds `[B,3,1024]`. The model invokes
ball query at `(N,Q,K,r)=(1024,512,32,0.2)` and `(512,128,64,0.4)`.
The strong baseline already uses `FarthestPointSamplingFused`; this table
therefore measures the incremental value of ball-query replacement.

| Batch | Torch NPU stage (ms) | Custom stage (ms) | Stage reduction | Baseline E2E (ms) | Custom E2E (ms) | E2E reduction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 37.291 | 0.646 | 98.27% | 49.109 | 12.463 | 74.62% |
| 4 | 58.572 | 1.037 | 98.23% | 71.592 | 13.977 | 80.48% |
| 8 | 80.580 | 1.699 | 97.89% | 95.717 | 16.369 | 82.90% |

Both call sites have zero index mismatches; maximum model-logit error is zero
and top-1 agreement is 100%. Raw evidence is in
`tests/model_e2e/results/pointnet2_ball_query_formal.json`.
