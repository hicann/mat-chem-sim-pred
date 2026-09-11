# API Reference

```cpp
uint64_t aclnnCsrAppnpPropagateFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels, int64_t iterations);

int32_t aclnnCsrAppnpPropagateFused(
    void* row_ptr, void* column_index, void* edge_weight, void* initial,
    void* output, int64_t nodes, int64_t edges, int64_t channels,
    int64_t iterations, float alpha, void* workspace,
    uint64_t workspace_size, void* stream);
```

Inputs are contiguous ND tensors: `row_ptr` INT32 `[nodes+1]`,
`column_index` INT32 `[edges]`, `edge_weight` FP32 `[edges]`, and `initial`
FP32 `[nodes,channels]`. Output is FP32 `[nodes,channels]`.

Validated limits are `1 <= channels <= 1024`, `1 <= iterations <= 64`, and
`0 <= alpha <= 1`. Null pointers, insufficient workspace, non-finite alpha,
invalid dimensions, or output/initial aliasing return `ACL_ERROR_INVALID_PARAM`.
Framework dispatch uses the custom path for at least 64 nodes and falls back for
unsupported dtype, layout, dimensions, or training-time APPNP dropout.
