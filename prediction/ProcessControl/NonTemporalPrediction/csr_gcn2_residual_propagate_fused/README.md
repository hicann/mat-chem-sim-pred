<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrGcn2ResidualPropagateFused

`CsrGcn2ResidualPropagateFused` implements the propagation and initial-residual
stage used by PyG `GCN2Conv`:
`R=(1-alpha)*P*X + alpha*X0`. The trained identity-mapping `addmm` remains a
native Cube operation. In the maintained Cora GCNII example this operator is
called once in each of 64 layers.

Inputs are contiguous INT32 CSR, FP32 edge weights, and equal-shape FP32
`current[N,C]` and `initial[N,C]` tensors. Output is FP32 `residual[N,C]`.
Inference dispatch supports valid CSR, finite weights, `0 <= alpha <= 1`,
`1 <= C <= 1024`, and INT32-representable sizes. Other dtypes, layouts,
normalization contracts, aliases, and autograd use the native path.

On Ascend 910B3, complete 64-layer GCNII latency on one/two/four Cora graph
copies changes from `22.460/39.360/73.973 ms` to `13.109/17.928/30.334 ms`,
reducing latency by `41.63%/54.45%/58.99%`. The per-layer stage is
`1.55x/1.79x/2.11x` faster and occupies `91.21%/94.83%/96.80%` of complete
model NPU kernel time. Both paths retain `79.20%` test accuracy.
