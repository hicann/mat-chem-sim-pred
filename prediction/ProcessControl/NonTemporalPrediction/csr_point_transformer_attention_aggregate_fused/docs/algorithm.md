<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For each edge `j->i`, the default PointTransformer message is
`delta_ij=pos_nn(p_i-p_j)`, `score_ij=a_i-a_j+delta_ij`, and
`message_ij=softmax_i(score_ij)*(v_j+delta_ij)`. Softmax is channelwise over
all edges in the destination CSR row.

The fusion consumes already projected `a_source`, `a_target`, and `value`, plus
edge-ordered `delta`. It removes the materialized score, normalized attention,
message, and scatter output tensors. A custom `attn_nn` changes the semantics
and is deliberately handled by fallback.
