<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For target `i`, source `j`, and head `h`, compute

`e_ijh = dot(query[i,h,:], key[j,h,:]) / sqrt(C)`,

apply stable softmax over target `i`'s CSR row, and return
`sum_j softmax(e_ijh) * value[j,h,:]`. Root skip and head concatenation/mean
are performed by the native TransformerConv layer around this stage.

The kernel fuses CSR source/target gathers, dot-product reduction, row
normalization, and weighted value reduction while avoiding edge-logit and
edge-value materialization.
