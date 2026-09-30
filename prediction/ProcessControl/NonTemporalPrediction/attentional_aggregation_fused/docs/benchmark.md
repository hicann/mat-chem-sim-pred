<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

## Environment and Method

| Item | Value |
|---|---|
| Accelerator | Ascend 910B3, physical device 6 (logical device 0 after visibility mapping) |
| CANN | 8.1.RC1.alpha001 |
| Framework | PyTorch 2.5, torch_npu 2.5.1, PyG 2.9.0 |
| Graph workload | PyG TUDataset(MUTAG), 188 graphs, 7 node features, 2 classes |
| Model | 3 x GCNConv(32) + AttentionalAggregation(Linear(32, 1)) + classifier |
| Checkpoint | SHA256 f45d91717e8d5bc9d180318dae56991574c311d834bb74b7b9eec6117aff03ef |
| Timing | synchronized resident NPU wall-clock median, warmup 10, repeat 50 |
| Excluded | H2D, D2H, checkpoint loading, and dataset preprocessing |

TorchAir is not available in this environment. The direct PyG path selects a
CPU scatter_reduce fallback, so it is retained as the maintained-source
semantic reference only and is deliberately not the performance baseline. The
reported baseline is the explicit all-NPU, algebraically equivalent dense-mask
stable-softmax pooling path using native torch_npu operations. On the same
resident GCN features, the direct PyG pooling call and this formulation agree
within 5.96e-8 for every measured shape. The full repeated PyG GCN path shows
a 3.61e-2 difference at the 188-graph workload on node202. This unrelated
environment-specific difference is not used to claim speedup: the custom path
is compared only with the all-NPU formulation, whose full-model output agrees
within 1.79e-7 at every shape.

## Full Model Results

| Graphs | Nodes | Edges | Stage baseline ms | Custom stage ms | Stage speedup | Strong E2E ms | Custom E2E ms | E2E reduction | Hotspot | Max model error | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 278 | 612 | 2.291 | 0.160 | 14.31x | 3.165 | 1.176 | 62.83% | 72.40% | 1.19e-7 | 0.6250 |
| 64 | 1,161 | 2,576 | 2.252 | 0.160 | 14.05x | 3.196 | 1.181 | 63.06% | 70.45% | 1.19e-7 | 0.7500 |
| 188 | 3,421 | 7,592 | 2.359 | 0.169 | 13.94x | 3.492 | 1.274 | 63.53% | 67.56% | 1.79e-7 | 0.7606 |
| 376 | 6,847 | 15,198 | 2.581 | 0.198 | 13.05x | 4.032 | 1.654 | 58.98% | 64.00% | 1.79e-7 | 0.7606 |

Strong E2E is the dense all-NPU equivalent baseline. Each measured batch has
identical baseline/custom predictions. The checkpoint test accuracy is 0.7632;
the per-batch accuracy values establish that the substitution preserves task
output.

## Dispatch Coverage

The component dispatch test uses 32 channels and 24 nodes per graph:

| Graphs | Nodes | Native all-NPU ms | Custom ms | Speedup | Max error |
|---:|---:|---:|---:|---:|---:|
| 4 | 96 | 2.482 | 0.175 | 14.16x | 1.19e-7 |
| 16 | 384 | 2.436 | 0.179 | 13.61x | 2.38e-7 |
| 64 | 1,536 | 2.430 | 0.182 | 13.37x | 7.15e-7 |
| 188 | 4,512 | 2.530 | 0.181 | 14.01x | 3.58e-7 |

For graph count below four, empty graphs, non-finite gates, or channels beyond
1024, the integration keeps the native path. No unsupported shape is counted
as a custom-path performance result.

Raw reproducibility artifacts:

~~~
tests/model_e2e/mutag_pyg_attentional_aggregation_final_20260801.json
tests/model_e2e/attentional_aggregation_dispatch_final_20260801.json
tests/model_e2e/benchmark_pyg_attentional_aggregation_e2e.py
tests/model_e2e/benchmark_shape_dispatch.py
~~~
