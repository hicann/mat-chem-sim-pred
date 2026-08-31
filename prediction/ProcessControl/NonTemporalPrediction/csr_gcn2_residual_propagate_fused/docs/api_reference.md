<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

OpDef `CsrGcn2ResidualPropagateFused` accepts INT32 `row_ptr[N+1]`, INT32
`source_index[E]`, FP32 `edge_weight[E]`, FP32 `current[N,C]`, FP32
`initial[N,C]`, and float attribute `alpha`. It returns FP32 `residual[N,C]`.

```cpp
uint64_t aclnnCsrGcn2ResidualPropagateFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels);
int32_t aclnnCsrGcn2ResidualPropagateFused(
    void* row_ptr, void* source_index, void* edge_weight, void* current,
    void* initial, void* output, int64_t nodes, int64_t edges, int64_t channels,
    float alpha, void* workspace, uint64_t workspace_size, void* stream);
```

The launcher rejects null pointers, output aliases, insufficient workspace,
invalid sizes or channels, and non-finite or out-of-range alpha. Framework
dispatch validates CSR structure, finite weights, contiguous FP32 tensors, and
the normalized-adjacency contract.
