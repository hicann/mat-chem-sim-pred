<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E

The audited benchmark uses
`HypergraphConv(use_attention=True, attention_mode="node")` on a Cora
neighborhood hypergraph with shared projections and head. It explicitly times
the faster resident eager decomposition. The synchronized samples, framework
versions, baseline choice, shape, and correctness are retained in
`model_e2e_hypergraph_c8_formal_run.json`.
