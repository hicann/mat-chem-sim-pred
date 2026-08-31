<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Release build includes the AIV kernel, ACL launcher, and linked OpDef: passed.
- Seven NumPy tests cover weighted propagation, empty rows, alpha boundaries,
  malformed CSR, invalid sources, mismatched weights, and non-finite values.
- CTest and ACL device smoke validate numerical output and invalid host
  dimensions on Ascend 910B3: passed.
- Three Cora scales cover stage, 64-layer E2E, checkpoint accuracy, agreement,
  and complete-model profiler attribution: passed.
- Python byte-compilation and Ruff checks: passed.
- Unsupported layouts, dtypes, shapes, normalization, and training fall back.
