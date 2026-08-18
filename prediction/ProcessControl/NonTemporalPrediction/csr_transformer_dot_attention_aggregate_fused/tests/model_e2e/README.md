<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E Reproduction

Train or load the maintained two-layer PyG TransformerConv Cora checkpoint with
`train_transformer_cora.py`, build for Ascend910B3, and run
`benchmark_transformer_cora_e2e.py` with copies `1 4 8`. The benchmark uses the
exact no-edge-feature TransformerConv path, compares official/resident/custom
component and complete-model latency, and reports task/prediction parity.

The checked-in JSON contains graph sizes, checkpoint hash, task accuracy,
synchronized raw samples, errors, and profiler evidence. Dataset and checkpoint
binaries are not stored in the repository.
