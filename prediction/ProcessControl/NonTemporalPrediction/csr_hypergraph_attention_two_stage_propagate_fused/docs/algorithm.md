<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The edge-major CSR supplies node incidences for each hyperedge. The first
kernel applies LeakyReLU to `[E,H]` incidence logits, computes a stable softmax
per hyperedge/head, stores normalized attention in workspace, and emits
`B_e * sum_i alpha_ie*x_i`.

The node-major CSR supplies incident hyperedges for each node. Its
`incidence_position` maps every node-major entry back to the edge-major
attention value. The second kernel emits
`D_i * sum_e alpha_ie*edge_feature_e`. Both kernels launch on the caller stream.

Workspace contains tiling metadata, `[M,H,C]` intermediate edge features, and
`[H,E]` normalized attention. This is the exact PyG node-attention reuse rule.
