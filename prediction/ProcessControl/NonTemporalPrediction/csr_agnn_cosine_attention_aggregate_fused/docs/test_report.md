# Test Report

- Fresh Release build of AIV kernel, ACL Host, OpDef, and smoke executable:
  passed on Ascend 910B3.
- ACL device smoke checks a non-trivial cosine softmax row and rejects an
  oversized segment: passed.
- Eight NumPy tests cover formula, isolated rows, dtype/shape, CSR endpoints,
  source range, and non-finite beta: passed.
- Exact three-scale component/E2E, checkpoint accuracy, prediction agreement,
  zero graph truncation, and complete-model Level1 profiler: passed.
- Same-host CPU 1/4/8/16-thread sweep, official NPU, custom resident NPU, and
  static-graph host-to-host E2E at 1/4/16 graph copies: passed. Host-to-host
  custom is `4.22x-6.76x` faster than CPU and keeps 81.80% test accuracy.
- Python byte compilation and Ruff: passed.
- Unsupported shapes, dtypes, layouts, malformed CSR, aliases, and autograd
  explicitly use native fallback.
- Complete-model TorchAir fullgraph is not established; no GE performance
  claim is made.
