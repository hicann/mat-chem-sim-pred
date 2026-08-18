<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For target `i`, source `j`, and head `h`, compute

`z_ijh = LeakyReLU(source[j,h,:] + target[i,h,:])`,
`e_ijh = dot(z_ijh, attention[h,:])`,

then apply stable softmax over the target CSR row and return
`sum_j softmax(e_ijh) * source[j,h,:]`.

The AIV kernel owns one target row, keeps logits in Unified Buffer, and fuses
the source/target gathers, dynamic activation, segment normalization, and
weighted channel reduction. Projection and layer bias remain native.
