<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrTagcnBasisK3Fused

`CsrTagcnBasisK3Fused` generates the four feature bases used by PyG
`TAGConv(K=3)`: `T0=X` and `Tk=L*Tk-1` for `k=1..3`. Three ordered AIV kernels
share one ACL call, so each propagation consumes the completed preceding basis
without host synchronization. The four trained Linear transforms remain native
Cube operations.

Inputs are contiguous INT32 `row_ptr[N+1]`, INT32 `source_index[E]`, FP32
`norm[E]`, and FP32 `features[N,C]`. Output is FP32 `basis[4,N,C]`. The custom
inference path supports `1 <= C <= 4096`, finite normalization weights, valid
CSR, and INT32-representable sizes. Unsupported layouts, dtypes, shapes,
normalization modes, and autograd use the native path.

On Ascend 910B3, complete two-layer PyG TAGCN latency on one/two/four Cora graph
copies changes from `2.324/4.700/9.788 ms` to `1.327/1.994/3.697 ms`, reducing
latency by `42.91%/57.57%/62.23%`. The isolated basis stage is
`2.16x/2.87x/3.64x` faster, while the complete-model NPU hotspot is
`87.72%/90.82%/91.60%`. Both paths retain `79.60%` test accuracy.
