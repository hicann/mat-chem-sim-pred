<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

## Completed Validation

| Check | Result |
|---|---|
| Release CMake build | Passed on Ascend 910B3 environment |
| Python reference tests | 5 passed |
| Python syntax compilation | Passed |
| ACL smoke example | Passed on physical device 6, exposed as logical device 0 |
| Static check | ruff check . passed |
| Component dispatch | 4 graph-count shapes, max error 7.15e-7 |
| Full model regression | 4 MUTAG batch sizes, max error 1.79e-7 |
| Prediction and metric regression | Exact prediction agreement; unchanged per-batch accuracy |

## Correctness Coverage

The Python tests cover a deterministic multi-graph example, randomized inputs,
non-uniform graph lengths, single-channel input, and invalid tensor contracts.
The ACL smoke example creates device tensors, launches the host API, copies the
result back, and compares it with the CPU reference.

## Model Regression

The test is the checkpointed PyG GCN plus AttentionalAggregation classifier
described in tests/model_e2e. Both stage and full-model outputs are compared
against the all-NPU stable-softmax reference. The measured full-model maximum
absolute error is at most 1.79e-7, and all classification predictions match
the reference.

The direct PyG aggregation output is checked as a source-level semantic
reference on the same GCN features: its maximum error against the all-NPU
stable-softmax reference is 5.96e-8. Its CPU fallback is excluded from
performance measurement. A node202 repeated-call GCN difference appears at
the 188-graph workload, so all performance and regression claims are explicitly
against the deterministic all-NPU equivalent baseline rather than that unstable
full-path execution.
