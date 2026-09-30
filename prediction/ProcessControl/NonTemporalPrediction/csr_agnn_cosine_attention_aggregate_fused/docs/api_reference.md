# API Reference

`aclnnCsrAgnnCosineAttentionAggregateFusedGetWorkspaceSize(nodes, edges,
channels, max_segment_size)` returns launch workspace bytes.

`aclnnCsrAgnnCosineAttentionAggregateFused(row_ptr, source_index, features,
normalized, output, nodes, edges, channels, max_segment_size, beta, workspace,
workspace_size, stream)` launches the forward operator.

- `row_ptr`: contiguous INT32 `[N+1]`, starting at zero and ending at `E`.
- `source_index`: contiguous INT32 `[E]`, each value in `[0,N)`.
- `features`, `normalized`, `output`: contiguous FP32 `[N,C]`.
- Limits: `N,E>0`, `C<=512`, maximum row size at most 2048, finite `beta`.

Framework dispatch validates tensor metadata and CSR content before launch.
Unsupported cases and autograd fall back to the maintained PyG native path.
The Host rejects null pointers, output aliases, invalid limits, non-finite
`beta`, and insufficient workspace with `ACL_ERROR_INVALID_PARAM`.
