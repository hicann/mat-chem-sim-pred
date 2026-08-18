<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrTransformerDotAttentionAggregateFused

`CsrTransformerDotAttentionAggregateFused` implements the inference message
stage of maintained PyG `TransformerConv` without edge features: scaled
multi-head dot-product attention, CSR segment softmax, and value aggregation.
The learned Q/K/V projections and root skip linear remain native. This is
distinct from GAT/GATv2 because the score is `Q_i dot K_j / sqrt(C)` and the
value projection is independent of the score projection.

The validated path supports contiguous INT32 CSR, FP32 `[N,H,C]` Q/K/V,
`H<=8`, `C<=32`, and row size at most 256. Edge-feature, beta, training
dropout, unsupported metadata/shapes, autograd, malformed CSR, and aliases use
the native TransformerConv path.
