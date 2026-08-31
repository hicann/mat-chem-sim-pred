<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

The ACLNN API accepts contiguous FP32 `position [N,3]` and contiguous int32
`idx_i`, `idx_j`, and `idx_k` tensors with the same triplet length. It writes a
distinct FP32 `angle [T]` tensor.

The workspace size is zero. The execution call launches on the caller's ACL
stream and retains no input or output pointer. The Python dispatch records all
tensors on the active NPU stream.

Unsupported dtypes, layouts, shapes, empty triplet sets, invalid indices, or
training use the maintained resident implementation.
