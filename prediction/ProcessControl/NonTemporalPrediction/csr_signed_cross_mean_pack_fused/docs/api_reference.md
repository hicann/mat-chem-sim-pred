<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

```cpp
uint64_t aclnnCsrSignedCrossMeanPackFusedGetWorkspaceSize(
    int64_t nodes, int64_t positive_edges, int64_t negative_edges,
    int64_t channels, int64_t max_segment_size);

int32_t aclnnCsrSignedCrossMeanPackFused(
    void* positive_row_ptr, void* positive_source_index,
    void* negative_row_ptr, void* negative_source_index, void* features,
    void* positive_inverse_degree, void* negative_inverse_degree,
    void* output, int64_t nodes, int64_t positive_edges,
    int64_t negative_edges, int64_t channels, int64_t max_segment_size,
    void* workspace, uint64_t workspace_size, void* stream);
```

The two row pointers and source-index arrays are contiguous INT32. `features`,
inverse degrees, and `output` are contiguous FP32 with shapes `[nodes, 2C]`,
`[nodes]`, `[nodes]`, and `[nodes, 4C]`.

Supported dimensions are positive INT32-sized node/edge counts,
`8 <= C <= 64` with `C % 8 == 0`, and
`1 <= max_segment_size <= 1024`. CSR monotonicity, terminal edge counts,
in-range source indices, inverse degrees, dtype, rank, contiguity, and aliasing
are contract requirements. The model wrapper dispatches to the maintained
native formula whenever the supported boundary is not met.
