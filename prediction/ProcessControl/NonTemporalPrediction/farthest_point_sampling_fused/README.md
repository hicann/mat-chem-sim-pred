<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# FarthestPointSamplingFused

`FarthestPointSamplingFused` deterministically selects point-cloud centers by
iterative farthest-point sampling, starting from index zero. Input is FP32
points `[B,N,3]`; output is INT32 indices `[B,K]`. It targets PointNet-style
point-cloud encoders.

`2 <= N <= 4096` and `1 <= K <= min(N,512)`. Build with
`cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3` and
run `python3 -m pytest tests -q`. The medium representative NPU subgraph changed
from `24.229` to `2.058 ms` (91.50% reduction). In a pretrained PointNet++ SSG
on real ModelNet40 samples, complete model latency changed from
`186.341/178.352/204.095 ms` to `50.261/70.015/94.873 ms` for B1/B4/B8
(73.03%/60.74%/53.52% reduction) with exact output parity. The model driver and
raw synchronized samples are under `tests/model_e2e/`.
