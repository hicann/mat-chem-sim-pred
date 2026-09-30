<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# feature: add AttentionalAggregationFused operator

## Type

- [x] New operator
- [x] Documentation and test coverage

## Background

This change adds AttentionalAggregationFused for the stable attention pooling
stage used by torch_geometric.nn.aggr.AttentionalAggregation.forward.
The validated consumer is a PyG graph classifier:

~~~
GCNConv(7, 32) -> GCNConv(32, 32) -> GCNConv(32, 32)
-> AttentionalAggregation(Linear(32, 1)) -> Linear(32, 2)
~~~

The native gate MLP remains unchanged. The custom operator replaces only the
per-graph stable softmax and weighted feature reduction. It therefore has a
clear API boundary and does not duplicate the model's learnable gate network.

## Scope

~~~
prediction/ProcessControl/NonTemporalPrediction/attentional_aggregation_fused/
~~~

The package includes Ascend C kernel and host launch code, a Python reference,
five correctness tests, an ACL smoke example, reproducible model and dispatch
benchmarks, and operator documentation.

## API and Semantics

~~~
aclnnAttentionalAggregationFused(
    graph_ptr: int32[G + 1],
    features: float32[N, C],
    gates: float32[N],
    pooled: float32[G, C])
~~~

For every graph g, the operator computes:

~~~
m_g = max_{i in g}(gates_i)
w_i = exp(gates_i - m_g) / sum_{j in g} exp(gates_j - m_g)
pooled_g = sum_{i in g} w_i * features_i
~~~

graph_ptr is a strictly increasing CSR-style graph offset array. The supported
custom path is finite FP32 input with 4 <= G, 1 <= C <= 1024, and non-empty
graphs. The integration must retain the native PyG path outside that contract.

## Model Evidence

The workload uses the cached PyG TUDataset(MUTAG): 188 molecular graphs,
7 node features, and 2 classes. The fixed 150/38 split checkpoint has SHA256
f45d91717e8d5bc9d180318dae56991574c311d834bb74b7b9eec6117aff03ef and
test accuracy 0.7632.

The measured stage is 64.00% to 72.40% of the resident full-model latency.
The PyG implementation may use a CPU scatter_reduce fallback on this software
stack, so it is used for semantic comparison only. The reported strong baseline
is an explicit all-NPU dense-mask softmax implementation. The direct PyG
pooling call matches that implementation within 5.96e-8 on the same features.
The repeated full PyG GCN path has a node202 difference at the 188-graph
workload; no performance or output-equivalence claim relies on that unstable
CPU-fallback execution.

On Ascend 910B3 with CANN 8.1.RC1.alpha001, PyTorch 2.5, torch_npu 2.5.1,
and PyG 2.9.0, synchronized resident inference (warmup 10, repeat 50, H2D/D2H
excluded) reports:

| Graphs | Nodes | Stage speedup | Strong E2E baseline (ms) | Custom E2E (ms) | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 16 | 278 | 14.31x | 3.165 | 1.176 | 62.83% |
| 64 | 1,161 | 14.05x | 3.196 | 1.181 | 63.06% |
| 188 | 3,421 | 13.94x | 3.492 | 1.274 | 63.53% |
| 376 | 6,847 | 13.05x | 4.032 | 1.654 | 58.98% |

The maximum full-model output error is 1.79e-7; prediction agreement and
per-batch task accuracy are unchanged for every measured shape.

## Validation

~~~
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
python -m pytest tests/test_attentional_aggregation_fused.py -q
python -m compileall -q .
ASCEND_RT_VISIBLE_DEVICES=6 ./build/examples/test_aclnn_attentional_aggregation_fused 0
ruff check .
~~~

The complete benchmark artifacts are under tests/model_e2e.

## Checklist

- [x] The change has one operator topic and no exploratory/rejected candidates.
- [x] Host launch, kernel, reference, tests, ACL smoke example, and documents are present.
- [x] The custom path is compared with a correct all-NPU baseline.
- [x] Model output regression and task metrics use a checkpointed MUTAG workload.
- [x] Dispatch coverage includes four graph-count shapes and a native fallback contract.
- [x] Build, correctness, ACL smoke, static checks, and git diff --check are run before submission.
