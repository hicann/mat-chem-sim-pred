<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

## ACLNN Host API

`aclnnPPFPointPairFeaturesFusedGetWorkspaceSize` validates the tensors and
returns a zero-byte workspace requirement plus an executor.

`aclnnPPFPointPairFeaturesFused` launches that executor on the supplied ACL
stream.

Inputs are contiguous FP32 `position [N,3]` and `normal [N,3]`, plus contiguous
int32 `source_index [E]` and `target_index [E]`. The output is a distinct FP32
tensor `[E,4]`. The caller owns every tensor and must keep it alive through
stream completion.

The Python dispatch accepts only supported, zero-offset NPU tensors. Empty
inputs, unsupported dtypes/layouts, invalid indices, and training paths use the
resident reference implementation.
