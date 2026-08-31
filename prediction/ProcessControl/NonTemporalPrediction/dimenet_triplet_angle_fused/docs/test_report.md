<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Reference tests cover collinear, orthogonal, degenerate, and random triplets.
- Dispatch tests cover supported inputs and all documented fallback guards.
- A clean Release build and the direct ACLNN CTest passed on Ascend 910B3.
- The ACLNN output is independently allocated, uses zero workspace, and is
  launched on the supplied current stream.
- Three real-QM9 model scales passed the recorded error threshold.
- TorchAir full-graph execution is not claimed.
