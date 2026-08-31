<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For stack `k`, target node `i`, and channel `c`, the operator computes

`Y[k,i,c] = activation(sum_e W[e] * X[k,src[e],c] + R[k,i,c] + B[k,c])`.

`e` spans the CSR row `[row_ptr[i], row_ptr[i+1])`. Activation is identity or
ReLU according to the required `relu` attribute. One AIV work item owns a
`(stack,target)` pair, so no atomic reduction is required. The channel vector
is padded only inside Unified Buffer; global input and output remain compact.

The operator does not fuse ARMA's learned projections. Those matrix
multiplications remain on Cube, while this operator removes gather, edge
scaling, scatter-add, two additions, and the internal activation launches.
