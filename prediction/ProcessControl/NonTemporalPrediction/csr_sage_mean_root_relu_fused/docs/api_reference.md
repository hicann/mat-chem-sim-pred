# API Reference

```cpp
uint64_t aclnnCsrSageMeanRootReluFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels);

int32_t aclnnCsrSageMeanRootReluFused(
    void* row_ptr, void* column_index, void* neighbor_features,
    void* root_features, void* output,
    int64_t nodes, int64_t edges, int64_t channels,
    void* workspace, uint64_t workspace_size, void* stream);
```

- `row_ptr`: INT32 `[nodes + 1]`, monotonic CSR offsets.
- `column_index`: INT32 `[edges]`, in-range source-node indices.
- `neighbor_features`: FP32 `[nodes, channels]`, bias-free left projection.
- `root_features`: FP32 `[nodes, channels]`, root projection plus left bias.
- `output`: FP32 `[nodes, channels]`.

The Host API rejects null pointers, non-positive dimensions, channels above
1024, and insufficient workspace with `ACL_ERROR_INVALID_PARAM`. Launch and
runtime errors are propagated unchanged. Input and output buffers must not
overlap.
