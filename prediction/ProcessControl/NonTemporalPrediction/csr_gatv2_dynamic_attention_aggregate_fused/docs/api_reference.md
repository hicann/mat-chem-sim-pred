<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

`aclnnCsrGatv2DynamicAttentionAggregateFusedGetWorkspaceSize(nodes, edges,
heads, channels, max_segment_size)` returns workspace bytes.

`aclnnCsrGatv2DynamicAttentionAggregateFused(row_ptr, source_index,
source_projected, target_projected, attention, output, nodes, edges, heads,
channels, max_segment_size, negative_slope, workspace, workspace_size, stream)`
launches the forward operator.

- CSR tensors are contiguous INT32; projections/output are FP32 `[N,H,C]`.
- `attention` is FP32 `[H,C]`; `H<=8`, `C<=32`, segment size `<=256`.
- `negative_slope` must be finite in `[0,1]`; output cannot alias inputs.

Framework metadata/content checks select this path only for the validated
contract. Edge-feature and training/autograd paths remain native.
