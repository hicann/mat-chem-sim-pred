<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Fresh Release build of AIV kernel, ACL Host, OpDef, and ACL smoke: passed.
- Smoke validates a one-edge positive path and rejects slope/segment limits.
- Seven NumPy tests cover dynamic activation/softmax and invalid dtype, shape,
  CSR, source, and slope inputs: passed.
- Cora checkpoint accuracy, three-scale component/E2E, model error,
  prediction agreement, no edge truncation, and Level1 profiler: passed.
- Python compile and Ruff: passed.
- Edge features, training dropout, unsupported metadata, aliases, malformed
  CSR, and autograd explicitly fall back to native GATv2Conv.
