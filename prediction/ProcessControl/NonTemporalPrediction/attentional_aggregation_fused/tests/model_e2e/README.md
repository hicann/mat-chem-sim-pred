<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model-Level Evidence

This directory records the reproducible model-level validation for
AttentionalAggregationFused.

benchmark_pyg_attentional_aggregation_e2e.py uses a PyG graph classifier with
the exact AttentionalAggregation.forward semantic contract. The gate MLP is
retained. The baseline replaces only the pooling implementation with an all-NPU
dense-mask stable-softmax formulation, because the direct PyG path can select
a CPU scatter_reduce fallback on this runtime. Direct PyG pooling on the same
GCN features is compared explicitly in the raw JSON. A node202 repeated-call
GCN difference at one full-path shape is also recorded; only the deterministic
all-NPU baseline is used for performance and regression claims.

The JSON files contain the workload identity, checkpoint SHA256, model
structure, timing methodology, component hotspot ratio, E2E timing, output
error, and task metrics. benchmark_shape_dispatch.py covers four supported
graph-count shapes. Native PyG remains the integration fallback for unsupported
shapes and non-finite inputs.
