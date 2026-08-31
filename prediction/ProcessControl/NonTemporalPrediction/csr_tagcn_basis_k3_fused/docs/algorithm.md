<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The operator implements the maintained PyG `TAGConv(K=3)` propagation basis for
a precomputed symmetrically normalized adjacency matrix in CSR form:

```text
T0 = X
T1 = L * T0
T2 = L * T1
T3 = L * T2
```

Stage 1 copies `T0` and computes `T1`. Stages 2 and 3 are queued on the same
stream and compute `T2` and `T3`. This preserves the device-wide dependencies
without host round trips. Each stage assigns complete CSR rows to AIV cores.

Graph normalization is cached outside both timed paths. The four Linear
transforms and bias remain native Cube operations and are not fused.
