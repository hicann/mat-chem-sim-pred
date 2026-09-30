<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E Reproduction

`benchmark_pyg_dgcnn_e2e.py` builds the maintained PyG
`SortAggregation(k=30)` DGCNN path, trains or loads the MUTAG checkpoint, and
compares official PyG, an equivalent resident torch_npu graph-convolution
path, and the custom operator. Every model E2E timing includes graph
convolutions and the classification head.

`benchmark_shape_dispatch.py` validates four direct component shapes and the
custom-path boundary. Immutable raw outputs are
`mutag_pyg_dgcnn_formal_20260801.json` and
`sort_aggregation_dispatch_20260801.json`.
