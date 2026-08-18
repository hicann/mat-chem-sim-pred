<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API reference

## Host API

```cpp
uint64_t aclnnCsrFilmModulatedMeanFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels,
    int64_t max_segment_size);

int32_t aclnnCsrFilmModulatedMeanFused(
    void* row_ptr, void* source_index, void* projected, void* beta,
    void* gamma, void* output, int64_t nodes, int64_t edges,
    int64_t channels, int64_t max_segment_size, int64_t apply_relu,
    void* workspace, uint64_t workspace_size, void* stream);
```

All tensor pointers refer to device memory. `workspace` stores the aligned
tiling record. The launch is asynchronous on the supplied ACL stream.

## Tensor shapes

| Name | Dtype | Shape | Meaning |
|---|---|---|---|
| `row_ptr` | int32 | `[N + 1]` | Target-sorted CSR offsets |
| `source_index` | int32 | `[E]` | Source node for every edge |
| `projected` | float32 | `[N, C]` | Native linear source projection |
| `beta` | float32 | `[N, C]` | Target FiLM shift |
| `gamma` | float32 | `[N, C]` | Target FiLM scale |
| `output` | float32 | `[N, C]` | Mean aggregated messages |

The host validates pointers, scalar ranges, workspace size, output aliasing,
`C <= 512`, `max_segment_size <= 2048`, and boolean `apply_relu`. Structural
CSR and finite-value validation belongs to the dispatch eligibility check.

`CsrFilmModulatedMeanFused` registers five inputs, one output, and required
boolean attribute `apply_relu` for Ascend 910B and Ascend 910_93.
