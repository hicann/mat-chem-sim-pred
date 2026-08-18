<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# API Reference

`aclnnPointBallQueryFused(points, queries, indices, counts, batch, point_count,
query_count, sample_count, radius, workspace, workspace_size, stream)` returns
`ACL_SUCCESS` or `ACL_ERROR_INVALID_PARAM`.

- `points`: contiguous FP32 `[B,N,3]`.
- `queries`: contiguous FP32 `[B,Q,3]`.
- `indices`: caller-allocated INT32 `[B,Q,K]`; unused positions are `-1`.
- `counts`: caller-allocated INT32 `[B,Q]` in `[0,K]`.
- `K=sample_count`, with `N<=8192`, `Q<=1024`, and `K<=128`.

Allocate device workspace with
`aclnnPointBallQueryFusedGetWorkspaceSize(B,N,Q,K)`. Inputs, outputs, workspace,
and stream must be non-null and resident on the selected device.
