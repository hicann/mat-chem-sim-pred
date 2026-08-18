<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

Given normalized adjacency entries `norm_ij`, the operator computes

`x1_i = sum_j norm_ij * x0_j`,

`x2_i = sum_j norm_ij * x1_j`,

`output_i = alpha0*x0_i + alpha1*x1_i + alpha2*x2_i`.

Stage 1 writes `x1` into workspace and initializes the output with layers 0/1.
Stage 2 propagates workspace values and accumulates layer 2. This removes two
edge-message materializations, two index-add launches, and the three-layer
stack/reduction used by native `LightGCN.get_embedding`.
