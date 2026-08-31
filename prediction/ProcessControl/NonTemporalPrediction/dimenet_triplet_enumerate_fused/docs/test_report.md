<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

Validation date: 2026-08-17. Runtime device: Ascend 910B3. Build target:
Ascend910B1. CANN: 8.1.RC1.alpha001.

| Gate | Result |
|---|---|
| Clean CMake/Ascend C build | PASS |
| ACLNN CTest | PASS, 1/1 |
| Python reference and dispatch | PASS, 5/5 |
| Random graph index semantics | PASS |
| Three runtime shapes | PASS, zero mismatched values |
| Output ownership | PASS, all outputs independent |
| Two current streams | PASS |
| 1,000-call stress | PASS, checksum stable, 512 B allocator growth |
| Real QM9 DimeNet++ E2E | PASS, 16.01%-19.70% reduction |

TorchAir's resident DimeNet path is not a valid fullgraph baseline in this
environment because the maintained model requires missing `pyg-lib` and
`torch-sparse` operations. A GE OPP package for the fixed-capacity custom
topology output remains pending; direct ACLNN model timing is the current
evidence boundary.
