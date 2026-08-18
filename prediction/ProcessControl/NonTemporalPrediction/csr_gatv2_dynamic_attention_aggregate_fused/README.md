<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrGatv2DynamicAttentionAggregateFused

`CsrGatv2DynamicAttentionAggregateFused` implements the inference message
stage of maintained PyG `GATv2Conv`: source/target projected feature sum,
LeakyReLU, per-target CSR softmax, and source projection aggregation. It is
semantically distinct from GAT's static source/target attention vectors because
the nonlinearity is applied before the learned attention dot product.

The validated path supports contiguous INT32 CSR and FP32
`[N,H,C]` projections, `[H,C]` attention, `H<=8`, `C<=32`, finite slope in
`[0,1]`, and row size at most 256. Edge features, unsupported layouts/shapes,
training dropout, autograd, malformed CSR, and aliases use native fallback.
