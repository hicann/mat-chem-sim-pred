<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Algorithm

For each `(batch, query)` row, the kernel scans source points in input order,
computes the FP32 squared Euclidean distance, and emits the first
`sample_count` indices whose distance is at most `radius^2`. Unused index slots
are `-1`; `counts` reports the number of emitted indices. The fused launch
replaces PointNet++ broadcasting, subtraction, reduction, comparison,
nonzero-index extraction, and slicing.

Rows are assigned independently across up to 40 AI cores. The host rejects
non-positive dimensions/radius, `point_count > 8192`, `query_count > 1024`, or
`sample_count > 128`. Callers should retain the framework path outside this
contract.
