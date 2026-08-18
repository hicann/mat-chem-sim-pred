<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Fresh Release build includes the AIV kernel, ACL launcher, linked OpDef, and
  device smoke executable: passed.
- Eight NumPy tests cover multi-head attention, stable softmax, isolated rows,
  malformed CSR, invalid sources, shape limits, non-finite values, and invalid
  negative slope: passed.
- CTest and ACL device smoke validate non-trivial two-neighbor softmax output
  and invalid Host parameters on Ascend 910B3: passed.
- One/four/eight complete Cora graph copies cover component latency, full-model
  E2E, task accuracy, prediction agreement, and zero edge truncation: passed.
- Conservative full-model profiler attribution excludes learned projections,
  layer bias, ELU, final reduction, and data movement: passed.
- Python byte-compilation and Ruff checks: passed.
- Unsupported dtypes, layouts, shapes, oversized rows, malformed graphs, and
  autograd fall back to the native path.
