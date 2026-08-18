<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

`aclnnCsrGatAttentionAggregateFusedGetWorkspaceSize(nodes, edges, heads,
channels, max_segment_size)` returns launch workspace bytes.

`aclnnCsrGatAttentionAggregateFused(row_ptr, source_index, projected,
attention_source, attention_target, output, nodes, edges, heads, channels,
max_segment_size, negative_slope, workspace, workspace_size, stream)` launches
the forward operator.

- `row_ptr`: contiguous INT32 `[N+1]`, starting at zero and ending at `E`.
- `source_index`: contiguous INT32 `[E]`, each source in `[0,N)`.
- `projected`, `output`: contiguous FP32 `[N,H,C]`.
- `attention_source`, `attention_target`: contiguous FP32 `[H,C]`.
- `negative_slope`: finite FP32 attribute in `[0,1]`.

Framework dispatch must verify tensor metadata, CSR content, and that no graph
row exceeds `max_segment_size`. The Host rejects null pointers, output aliases,
invalid limits, invalid slope, and insufficient workspace.
