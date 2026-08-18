<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Fresh Release build of AIV kernel, ACL Host, OpDef, and ACL smoke: passed.
- Smoke validates one-edge dot attention and rejects an oversized segment.
- Seven NumPy tests cover scaled dot attention, weighted values, dtype/shape,
  CSR endpoint/source, and channel limits: passed.
- Cora checkpoint accuracy, three-scale component/E2E, prediction agreement,
  no truncation, and complete-model Level1 profiler: passed.
- Python compile and Ruff: passed.
- Edge-feature/training/beta paths, unsupported metadata, aliases, malformed
  CSR, and autograd explicitly fall back to native TransformerConv.
