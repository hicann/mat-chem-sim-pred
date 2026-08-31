<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The operator implements the PyG `ChebConv(K=3)` recurrence for a precomputed,
scaled normalized Laplacian in CSR form:

```text
T0 = X
T1 = L * T0
T2 = 2 * L * T1 - T0
```

Stage 1 copies `T0` and computes `T1` by weighted CSR reduction. Stage 2 is
queued on the same stream, consumes `T1`, and computes `T2`. This provides the
required device-wide dependency without a host round trip. Each stage assigns
complete CSR rows to AIV cores and uses channel-sized UB buffers.

PyG graph normalization is performed once outside the timed inference region,
equally for baseline and custom paths. Linear transforms and bias addition are
not fused so they continue to execute on Cube.
