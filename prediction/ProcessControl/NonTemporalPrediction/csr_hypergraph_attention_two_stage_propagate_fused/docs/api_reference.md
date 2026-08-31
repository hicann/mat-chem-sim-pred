<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# API reference

The workspace query takes `(N,M,E,H,C,max_edge_size,max_node_degree)`. Launch
inputs are edge-major CSR (`edge_row_ptr`, `node_index`, `edge_scale`),
node-major CSR (`node_row_ptr`, `edge_index`, `incidence_position`,
`node_scale`), same-dtype FP32/FP16/BF16 `features [N,H,C]` and
`attention_logits [E,H]`, and a preallocated output `[N,H,C]`. A `dtype` scalar
follows the seven shape scalars: `0` (FP32), `1` (FP16), or `2` (BF16).

All five index arrays are contiguous int32. Scales and features are contiguous
and same dtype. Softmax and message accumulation use FP32. Limits: `H<=4`,
`C<=32`, both segment maxima `<=512`, and slope in `[0,1]`.
The adapter validates the two CSR endpoint sets and mapping range. Unsupported
inputs run PyG. Output/workspace cannot alias any input or each other.
