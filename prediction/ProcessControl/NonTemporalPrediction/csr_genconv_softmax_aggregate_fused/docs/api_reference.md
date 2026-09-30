# API Reference

## OpDef

Operator name: `CsrGenConvSoftmaxAggregateFused`

Inputs:

- `row_ptr`: contiguous INT32 `[N + 1]`
- `source_index`: contiguous INT32 `[E]`
- `features`: contiguous FP32 `[N, C]`

Attribute:

- `temperature`: finite floating-point inverse temperature

Output:

- `output`: FP32 `[N, C]`

## ACL C Interface

```cpp
uint64_t aclnnCsrGenConvSoftmaxAggregateFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels, float temperature);

int32_t aclnnCsrGenConvSoftmaxAggregateFused(
    void* row_ptr, void* source_index, void* features, void* output,
    int64_t nodes, int64_t edges, int64_t channels, float temperature,
    void* workspace, uint64_t workspace_size, void* stream);
```

The launch returns `ACL_ERROR_INVALID_PARAM` for null pointers, insufficient
workspace, dimensions outside the documented range, or a non-finite
temperature. Input tensors must be contiguous. Framework dispatch code must
materialize a contiguous tensor or fall back before calling this raw-pointer
interface.
