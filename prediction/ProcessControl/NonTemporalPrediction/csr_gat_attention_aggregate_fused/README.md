<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrGatAttentionAggregateFused

`CsrGatAttentionAggregateFused` implements the complete sparse attention stage
of PyG `GATConv`: source/target attention scoring from projected node features,
LeakyReLU, stable CSR segment softmax, and weighted neighbor aggregation. It
avoids materializing edge values and logits. The maintained `examples/gat.py`
Cora model invokes this stage in both GAT layers.

Inputs use contiguous INT32 CSR, FP32 `projected[N,H,C]`, and FP32
`attention_source[H,C]`/`attention_target[H,C]`. Forward dispatch supports
`H<=8`, `C<=32`, segment size at most 256, valid non-truncated CSR, finite
`0<=negative_slope<=1`, and non-aliasing output. Other shapes, dtypes, layouts,
graphs, and autograd use the native path.

Formal component, complete-model, task-accuracy, profiler, and ACL evidence is
recorded in `docs/benchmark.md` and `docs/test_report.md`.
