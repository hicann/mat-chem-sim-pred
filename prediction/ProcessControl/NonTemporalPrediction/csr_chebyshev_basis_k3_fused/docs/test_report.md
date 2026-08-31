<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- NumPy reference covers weighted recurrence, empty rows, negative weights,
  malformed CSR, invalid sources, mismatched norm length, and non-finite norm.
- Release build includes two AIV kernels, host launcher, and OpDef.
- Seven NumPy reference and validation tests passed.
- CTest passed 1/1; direct ACL smoke validates both ordered kernel stages.
- Three real Cora scales pass component and complete-model accuracy checks.
- Complete-model Level1 NPU profiles pass the 15% hotspot gate at all scales.
- Framework fallback applies to unsupported layouts, dtypes, channels,
  normalization contracts, and training.
- Python compile and Ruff checks passed with no suppressions.
