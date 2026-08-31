<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The operator implements the pre-`addmm` stage in maintained PyG `GCN2Conv`:

```text
propagated = P * current
residual = (1 - alpha) * propagated + alpha * initial
```

`P` is a precomputed symmetrically normalized adjacency matrix with self-loops,
stored in target-row CSR. Each AIV core owns complete target rows, accumulates
weighted source features, and combines the matching initial feature row. The
layer-specific `beta` and trained weight matrix stay in native `torch.addmm`.
