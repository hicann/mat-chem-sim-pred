<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

```cpp
uint64_t aclnnDimeNetTripletEnumerateFusedGetWorkspaceSize(
    int64_t nodes, int64_t edges, int64_t capacity);

int32_t aclnnDimeNetTripletEnumerateFused(
    void* row_ptr, void* source_index,
    void* idx_i, void* idx_j, void* idx_k,
    void* idx_kj, void* idx_ji, void* counts,
    int64_t nodes, int64_t edges, int64_t capacity,
    void* workspace, uint64_t workspace_size, void* stream);
```

All data tensors are contiguous `int32`. `row_ptr` has `nodes + 1` elements,
`source_index` has `edges`, each triplet output has `capacity`, and `counts`
has two elements. The host rejects null pointers, aliasing inputs/outputs,
undersized workspace, non-positive sizes, and sizes outside signed INT32.

The PyTorch binding registers
`cann_prediction::dimenet_triplet_enumerate_fused` and returns a six-tensor
tuple. It launches on `torch.npu.current_stream()` and records input, output,
and workspace lifetimes on that stream.
