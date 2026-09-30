# API reference

`aclnnCsrSuperGatDotAttentionAggregateFusedGetWorkspaceSize(N,E,H,C,max_degree)`
returns the required workspace bytes.

`aclnnCsrSuperGatDotAttentionAggregateFused(row_ptr, source_index, projected,
output, N, E, H, C, max_degree, dtype, negative_slope, workspace, workspace_size,
stream)` launches asynchronously on `stream`.

`row_ptr` and `source_index` are contiguous int32. `projected` and `output` are
same-dtype contiguous `[N,H,C]` tensors. `dtype` is `0` (FP32), `1` (FP16), or
`2` (BF16); softmax and message accumulation use FP32. Limits: `N,E>0`, `H<=8`,
`C<=64`, `max_degree<=512`, and `0<=negative_slope<=1`. The adapter must
validate CSR endpoints/indices and fall back for unsupported dtype/layout,
zero-edge, or oversized rows. Its `torch.library` binding registers an autograd
recomputation for training, but no dedicated AscendC backward kernel. Output
and workspace must not alias inputs or each other.
