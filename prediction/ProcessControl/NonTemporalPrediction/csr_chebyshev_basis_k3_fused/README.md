<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrChebyshevBasisK3Fused

`CsrChebyshevBasisK3Fused` generates the three feature bases used by PyG
`ChebConv(K=3)`: `T0=X`, `T1=LX`, and `T2=2L(T1)-X`. Two ordered AIV kernels
share one ACL call, so the second sparse propagation reads the completed first
basis without host synchronization. The ChebConv Linear layers remain native
Cube operations.

Inputs are contiguous INT32 `row_ptr[N+1]`, INT32 `source_index[E]`, FP32
`norm[E]`, and FP32 `features[N,C]`. Output is FP32 `basis[3,N,C]`. The custom
inference path supports `1 <= C <= 4096`, finite normalization weights, valid
CSR, and INT32-representable sizes. Unsupported layouts, types, shapes, graph
normalization modes, and autograd use the native path.

On Ascend 910B3, complete two-layer PyG ChebNet latency on one/two/four Cora
graph copies changes from `1.938/3.808/8.159 ms` to `1.069/1.598/2.953 ms`, a
`44.87%/58.05%/63.81%` reduction. The isolated basis stage is
`2.10x/2.99x/3.64x` faster, and a complete-model NPU profile attributes
`87.99%/92.43%/92.87%` of kernel time to that stage. Both paths retain
`73.80%` test accuracy. See `docs/benchmark.md` for the evidence boundary.
