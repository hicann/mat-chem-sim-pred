<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API Reference

## Host API

~~~
uint64_t aclnnAttentionalAggregationFusedGetWorkspaceSize(
    int64_t nodes, int64_t graphs, int64_t channels);

int32_t aclnnAttentionalAggregationFused(
    void* graphPtr, void* features, void* gates, void* output,
    int64_t nodes, int64_t graphs, int64_t channels,
    void* workspace, uint64_t workspaceSize, void* stream);
~~~

## Tensors

| Tensor | Dtype | Shape | Meaning |
|---|---|---|---|
| graphPtr | int32 | [G + 1] | CSR graph offsets |
| features | float32 | [N, C] | Node features |
| gates | float32 | [N] | Scalar gate logits from the native gate MLP |
| pooled | float32 | [G, C] | Per-graph pooled features |

The caller passes device pointers and explicit dimensions. The required logical
shapes are graphPtr [G + 1], features [N, C], gates [N], and output [G, C].
The dimension relations are graphPtr[0] = G + 1, features[0] = gates[0] = N,
and output = [G, C].

## Errors and Dispatch

The host API returns ACL_ERROR_INVALID_PARAM for null pointers, insufficient
workspace, non-positive dimensions, graph count above node count, channel count
above 1024, or dimensions beyond the uint32_t tiling range. The C ABI does not
carry tensor descriptors, so dtype and logical-shape checks are part of the
model integration. Conditions that require device input values (graphPtr
monotonicity, no empty graph, and finite gate values) are also checked before
dispatch. Unsupported inputs must use the native PyG path rather than invoking
this operator.

The current kernel uses an aligned device-resident tiling record as workspace;
GetWorkspaceSize returns its required size.
