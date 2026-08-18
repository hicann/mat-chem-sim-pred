<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For target `i`, source `j`, and head `h`, the operator computes

`e_ijh = LeakyReLU(<X[j,h], a_src[h]> + <X[i,h], a_dst[h]>)`,

then applies a numerically stable softmax over the target's CSR row and returns
`sum_j softmax(e_ijh) * X[j,h,:]`.

One AIV work item owns one target node. It scans the CSR row once to compute
scores and their maximum, applies vector exponential in Unified Buffer, and
scans again for the weighted channel reduction. This fuses the two node
gathers, score reductions, activation, segment softmax, and aggregation while
leaving the learned projection and layer bias on native Cube/vector kernels.
