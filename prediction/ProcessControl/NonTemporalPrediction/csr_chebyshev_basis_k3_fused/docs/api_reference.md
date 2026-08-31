<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

OpDef `CsrChebyshevBasisK3Fused` accepts `row_ptr` INT32 `[N+1]`,
`source_index` INT32 `[E]`, `norm` FP32 `[E]`, and contiguous `features` FP32
`[N,C]`. It returns `basis` FP32 `[3,N,C]`.

```cpp
uint64_t aclnnCsrChebyshevBasisK3FusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t channels);
int32_t aclnnCsrChebyshevBasisK3Fused(
    void* row_ptr, void* source_index, void* norm, void* features, void* basis,
    int64_t nodes, int64_t edges, int64_t channels, void* workspace,
    uint64_t workspace_size, void* stream);
```

The launcher rejects null pointers, insufficient workspace, non-positive or
INT32-unrepresentable graph sizes, and channels outside `[1,4096]`. Dispatch
must validate finite normalization weights and contiguous FP32 tensors.
