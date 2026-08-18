<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrLightgcnK2WeightedSumFused

`CsrLightgcnK2WeightedSumFused` implements the complete `K=2` LightGCN
inference propagation: two symmetric-normalized CSR passes and the learned or
uniform weighted sum of layers 0, 1, and 2. It replaces the maintained PyG
`LightGCN.get_embedding` propagation while leaving pair scoring native.

The forward path supports contiguous INT32 CSR, FP32 edge norms and `[N,C]`
features with `C<=512`, finite layer weights, non-empty valid CSR, and
non-aliasing output. Unsupported dtypes, layouts, shapes, malformed graphs, or
autograd use the native PyG path.
