<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For hidden state `[positive, negative]`, positive edge CSR `P`, and negative
edge CSR `N`, the operator emits:

```text
[
  mean_P(positive),
  mean_N(negative),
  mean_P(negative),
  mean_N(positive)
]
```

This is the exact neighborhood pack consumed by maintained PyG
`SignedConv(first_aggr=False)`. The resident path launches four gathers, four
scatter/index additions, four degree multiplications, and a concatenation. The
custom kernel owns complete destination rows, accumulates all four outputs in
local memory, and writes the packed result directly without atomics or
intermediate node tensors.

Nodes are distributed across at most 40 AI cores. Channels are FP32 and a
multiple of eight so all four local-buffer segments satisfy Ascend alignment.
