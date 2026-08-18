<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Fresh Release build of two AIV kernels, ACL Host, OpDef, and smoke: passed.
- ACL smoke executes both hops and rejects non-finite layer weights: passed.
- Seven NumPy tests cover exact K2 semantics, dtype/shape, CSR endpoints,
  source range, and non-finite attributes: passed.
- Official MovieLens split, real checkpoint, Recall@20, three-scale component
  and E2E, score parity, and conservative complete-model profiler: passed.
- Python byte compilation and Ruff: passed.
- Unsupported inputs, aliases, malformed CSR, and autograd use native fallback.
