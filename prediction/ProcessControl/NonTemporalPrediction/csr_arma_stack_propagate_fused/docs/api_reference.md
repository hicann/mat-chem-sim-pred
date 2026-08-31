<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

`aclnnCsrArmaStackPropagateFusedGetWorkspaceSize(nodes, edges, stacks,
channels)` returns workspace bytes for the launch tiling.

`aclnnCsrArmaStackPropagateFused(row_ptr, source_index, edge_weight, projected,
root, bias, output, nodes, edges, stacks, channels, relu, workspace,
workspace_size, stream)` launches the forward operator.

- `row_ptr`: contiguous INT32 `[N+1]`, starting at zero and ending at `E`.
- `source_index`: contiguous INT32 `[E]`, each value in `[0,N)`.
- `edge_weight`: contiguous FP32 `[E]`.
- `projected`, `root`, `output`: contiguous FP32 `[K,N,C]`.
- `bias`: contiguous FP32 `[K,C]`.
- `relu`: integer ABI value `0` or `1`, exposed as a required Bool OpDef attr.

The host rejects null pointers, output aliases, invalid sizes, integer
overflow, unsupported stack/channel bounds, invalid ReLU values, and
insufficient workspace. Tensor metadata and CSR content validation belong to
the framework dispatch before launch.
