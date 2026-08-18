<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test report

Date: 2026-08-02

## Functional coverage

- FiLM scale/shift, optional message ReLU, and CSR mean
- Single-edge and empty-row behavior
- Random direct-loop comparison
- Malformed CSR start, monotonicity, final offset, and source indices
- Dtype, shape, activation type, training, non-contiguous, non-finite,
  channel, and fan-in fallback
- Host scalar validation and real ACL device launch

`pytest`: 11 passed. `ctest`: one ACL smoke test passed on Ascend 910B3.
Release AscendC kernel, Host, and OpDef libraries built successfully. Ruff and
Python byte-code compilation passed.

## Model regression

The exact maintained PyG FiLM PPI topology was trained for 50 epochs. The
checkpoint reaches 0.8650 validation and 0.8866 test micro-F1. Across two to
eight real PPI graphs, all binary predictions match official PyG. Maximum
component error is `7.6294e-6`, and maximum normalized E2E error is `2.56e-7`.

## Performance gate

The operator passes the `>=1.3x` component gate at all sizes: `4.18x`,
`5.82x`, and `6.65x`. All four model stages enabled together reduce E2E
latency by `60.29%`, `69.05%`, and `72.31%` against official PyG.
