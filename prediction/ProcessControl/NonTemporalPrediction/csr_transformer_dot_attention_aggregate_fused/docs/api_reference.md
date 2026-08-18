<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

`aclnnCsrTransformerDotAttentionAggregateFusedGetWorkspaceSize(nodes, edges,
heads, channels, max_segment_size)` returns launch workspace bytes.

`aclnnCsrTransformerDotAttentionAggregateFused(row_ptr, source_index, query,
key, value, output, nodes, edges, heads, channels, max_segment_size, workspace,
workspace_size, stream)` launches the forward stage.

- CSR tensors are contiguous INT32; Q/K/V/output are FP32 `[N,H,C]`.
- Limits: `N,E>0`, `H<=8`, `C<=32`, maximum row size `<=256`.
- Output must not alias query, key, or value; Host rejects invalid metadata and
  workspace with `ACL_ERROR_INVALID_PARAM`.

Only the validated no-edge-feature, inference path dispatches here. Edge
features, beta, dropout, training, unsupported shapes, and autograd use native
TransformerConv.
