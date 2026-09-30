<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

```cpp
uint64_t aclnnSortAggregationTopkFusedGetWorkspaceSize(
    int64_t nodes, int64_t graphs, int64_t channels, int64_t top_k);

int32_t aclnnSortAggregationTopkFused(
    void* graph_ptr, void* features, void* output,
    int64_t nodes, int64_t graphs, int64_t channels, int64_t top_k,
    void* workspace, uint64_t workspace_size, void* stream);
```

- `graph_ptr`: INT32 `[graphs + 1]`, monotonic offsets with first value zero
  and final value `nodes`.
- `features`: FP32 `[nodes, channels]`; values in the final channel must be
  finite.
- `output`: FP32 `[graphs, top_k * channels]`.

The Host API rejects null pointers, non-positive dimensions, `channels` above
1024, `top_k` above 128, and insufficient workspace with
`ACL_ERROR_INVALID_PARAM`. Launch and runtime errors are propagated. Input and
output buffers must not overlap. Semantic validation of `graph_ptr` contents
remains the caller's responsibility, matching other raw CSR/segment APIs.
