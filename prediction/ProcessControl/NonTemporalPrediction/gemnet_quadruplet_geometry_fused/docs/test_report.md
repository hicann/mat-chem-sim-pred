<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- NumPy reference tests cover bond angles, projected torsion, degenerate
  vectors, and invalid topology.
- A clean Release build and the direct ACLNN CTest passed on Ascend 910B3.
- NPU smoke tests compare all three outputs with the official resident formula.
- Workspace is zero bytes; output ownership and current-stream recording are
  enforced by the binding.
- Independent real-QM9 model runs passed with maximum output error `7.63e-06`.
- TorchAir full-graph execution was unavailable for the official source path
  and is not claimed.
