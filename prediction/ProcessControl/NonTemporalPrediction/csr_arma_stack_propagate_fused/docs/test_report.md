<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Fresh Release build includes the AIV kernel, ACL launcher, linked OpDef, and
  device smoke executable: passed.
- Eight NumPy tests cover weighted stacks, both activation modes, empty rows,
  malformed CSR, invalid sources, shape mismatches, and non-finite values.
- CTest and ACL device smoke validate numerical output and invalid Host
  parameters on Ascend 910B3: passed.
- Three Cora scales cover component, full-model E2E, task accuracy, prediction
  agreement, and conservative complete-model profiler attribution: passed.
- Official PyG and staged checkpoint forward parity: zero error, passed.
- Python byte-compilation and Ruff checks: passed.
- Unsupported dtypes, layouts, shapes, normalization, and training fall back.
