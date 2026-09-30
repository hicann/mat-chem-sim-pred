<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# AttentionalAggregationFused

AttentionalAggregationFused implements the stable softmax-weighted graph
pooling portion of PyG torch_geometric.nn.aggr.AttentionalAggregation.forward.

It is validated in a graph-level MUTAG classifier:

~~~
3 x GCNConv(32) -> AttentionalAggregation(Linear(32, 1)) -> Linear(32, 2)
~~~

The gate MLP remains native. This operator accepts its scalar output and fuses
the per-graph max, exponential normalization, and weighted reduction.

## Contract

~~~
graph_ptr  int32   [G + 1]
features   float32 [N, C]
gates      float32 [N]
pooled     float32 [G, C]
~~~

graph_ptr must be strictly increasing from 0 to N; each graph must contain at
least one node. The custom FP32 path supports finite values, G >= 4, and
1 <= C <= 1024. Callers use the native PyG implementation for unsupported
shapes or non-finite gates.

See docs/algorithm.md and docs/api_reference.md for the full semantic and API
contract.

## Build and Test

~~~
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
python -m pytest tests/test_attentional_aggregation_fused.py -q
python -m compileall -q .
ASCEND_RT_VISIBLE_DEVICES=6 ./build/examples/test_aclnn_attentional_aggregation_fused 0
~~~

## Measured Model Result

On Ascend 910B3, an all-NPU equivalent baseline for the full PyG classifier
was compared with the custom operator at 16, 64, 188, and 376 MUTAG graphs.
The stage speeds up by 13.05x to 14.31x and full-model latency falls by
58.98% to 63.53%, with maximum model error 1.79e-7 and unchanged predictions.

The full environment, timing method, baseline rationale, and raw artifacts are
in docs/benchmark.md and tests/model_e2e.
