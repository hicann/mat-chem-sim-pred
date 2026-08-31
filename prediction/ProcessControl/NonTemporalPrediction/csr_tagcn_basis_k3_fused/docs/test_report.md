<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- NumPy reference covers weighted propagation, empty rows, negative weights,
  malformed CSR, invalid sources, mismatched norm length, and non-finite norm.
- Release build includes three ordered AIV kernels, host launcher, and OpDef.
- Seven NumPy reference and validation tests passed.
- CTest passed 1/1; ACL smoke validates all stages and invalid host dimensions.
- Three real Cora scales cover component, hotspot, E2E, accuracy, and agreement.
- All scales exceed the 15% complete-model hotspot and 1.3x component gates.
- Framework fallback applies to unsupported layouts, dtypes, channels,
  normalization contracts, and training.
- Python compile and Ruff checks passed with no suppressions.
