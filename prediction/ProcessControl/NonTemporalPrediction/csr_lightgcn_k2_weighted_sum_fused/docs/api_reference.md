<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

`aclnnCsrLightgcnK2WeightedSumFusedGetWorkspaceSize(nodes, edges, channels)`
returns tiling plus `[N,C]` temporary workspace bytes.

`aclnnCsrLightgcnK2WeightedSumFused(row_ptr, source_index, norm, features,
output, nodes, edges, channels, alpha0, alpha1, alpha2, workspace,
workspace_size, stream)` launches both propagation stages.

- `row_ptr`: contiguous INT32 `[N+1]`; `source_index`: INT32 `[E]`.
- `norm`: contiguous FP32 `[E]`; `features`, `output`: FP32 `[N,C]`.
- Limits: `N,E>0`, `C<=512`, finite `alpha0/alpha1/alpha2`.

Framework dispatch verifies monotonic CSR, endpoints, source ranges, metadata,
and non-aliasing output. Unsupported cases and autograd use native fallback.
