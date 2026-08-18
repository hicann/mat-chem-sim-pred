<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Model E2E Reproduction

Train or load the maintained two-layer PyG GATv2 Cora checkpoint with
`train_gatv2_cora.py`, build the package for Ascend910B3, and run
`benchmark_gatv2_cora_e2e.py` with copies `1 4 8`. The benchmark compares the
official PyG scatter path, an exact resident padded NPU path, and this CSR
fusion, reporting component and full-model latency/error plus task agreement.

The checked-in JSON also records checkpoint hash, test accuracy, graph sizes,
and raw synchronized samples. CPU fallback in official scatter is reported
explicitly rather than hidden.
