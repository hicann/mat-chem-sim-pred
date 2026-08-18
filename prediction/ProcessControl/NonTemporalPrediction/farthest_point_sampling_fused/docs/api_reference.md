<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# API

`aclnnFarthestPointSamplingFused(points, sample_indices, batch, point_count,
sample_count, workspace, workspace_size, stream)` returns `ACL_SUCCESS` or
`ACL_ERROR_INVALID_PARAM`.

`points` is FP32 `[B,N,3]` and `sample_indices` is INT32 `[B,K]`. The API
requires `2 <= N <= 4096` and `1 <= K <= min(N,512)`. Allocate workspace with
`aclnnFarthestPointSamplingFusedGetWorkspaceSize` and pass an ACL stream.
