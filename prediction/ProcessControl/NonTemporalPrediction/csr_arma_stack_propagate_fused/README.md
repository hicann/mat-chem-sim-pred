<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrArmaStackPropagateFused

`CsrArmaStackPropagateFused` implements the repeated propagation stage in PyG
`ARMAConv`: weighted CSR aggregation over `K` parallel stacks, root and bias
addition, and optional ReLU. Linear projections remain native Cube operations.
The maintained `examples/arma.py` Cora model invokes this stage four times.

Inputs use contiguous INT32 CSR, FP32 edge weights, FP32
`projected[K,N,C]`, equal-shape FP32 `root`, and FP32 `bias[K,C]`. The forward
dispatch supports `1 <= K <= 64`, `1 <= C <= 1024`, valid normalized CSR, and
non-aliasing output. Other dtypes, layouts, shapes, normalization contracts,
and autograd use the native path.

On Ascend 910B3, complete-model latency for one/two/four Cora graph copies
changes from `3.137/5.431/10.049 ms` to `2.039/3.536/6.537 ms`, reducing
latency by `35.01%/34.90%/34.95%`. The isolated stage is
`1.41x/1.46x/1.49x` faster and conservatively occupies
`76.95%/81.85%/85.40%` of complete-model NPU kernel time. Both paths retain
`79.50%` test accuracy.
