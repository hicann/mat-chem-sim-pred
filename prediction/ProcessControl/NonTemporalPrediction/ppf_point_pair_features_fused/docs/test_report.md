<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- NumPy reference tests cover the feature order, degenerate geometry, and
  validation failures.
- Dispatch tests cover supported execution and shape, dtype, layout, index,
  empty-input, and training fallbacks.
- A clean Release build and ACLNN CTest passed on Ascend 910B3.
- Current-stream tensor recording and independent output ownership are checked
  by the binding smoke test.
- Real ModelNet10 model outputs agree within `4.47e-08` at the largest audited
  scale.
- TorchAir full-graph execution was not validated for this ACLNN geometry path;
  no graph-capture support claim is made.
