# Test Report

## Static and Reference Tests

- NumPy reference covers feature-wise softmax, numerical stability, empty
  rows, negative temperature, malformed CSR, invalid sources, and non-finite
  temperature.
- Python compile and Ruff checks cover the package benchmark and reference
  code.
- The OpDef registers three inputs, one output, and the temperature attribute
  for Ascend 910B targets.

## Device Tests

- Release build on Ascend 910B3.
- CTest ACL smoke through the exported `aclnn` interface.
- Direct ACL smoke compares the AIV result with an independent host reference.
- Host negative cases reject INT32-unrepresentable node counts and non-finite
  temperatures.
- The value-gate wrapper prepares persistent workspace and caller-owned output
  buffers for every trained DeeperGCN layer before asynchronous timing.

## Acceptance Result

- Three meaningful Cora graph scales pass.
- The historical component result against global padding is retained only for
  reproduction; that baseline expands the source slots by `34.50x`.
- Relative to the accepted `1.309x` degree-bucketed all-NPU baseline, complete
  model latency reduction is `60.02%-77.81%`.
- Custom host-to-host latency is `5.924/7.095/10.595 ms` for one/two/four
  disjoint Cora graphs; the same-host official PyG CPU minima are
  `497.352/573.072/698.806 ms` after a `1/4/8/16` thread sweep.
- Maximum complete-model error is `3.34e-6`, prediction agreement is `100%`,
  and CPU/custom test accuracy is `70.10%`.
- Unsupported layouts, dtypes, edge-message variants, and training use the
  native fallback in framework integration.

## CodeCheck Remediation Validation (2026-09-20)

- `python -m pytest tests -q`: 16 passed on Windows, Python 3.11.9,
  CPU PyTorch 2.14.0 and PyG 2.8.0.post1. This includes the original seven NumPy
  reference cases and nine benchmark-helper regression cases.
- Operator-scoped pre-commit hooks pass, including clang-format, Ruff, JSON,
  whitespace, private-key detection and spelling checks. Repository-wide
  pre-commit has unrelated baseline failures and is not claimed as passing.
- Python compile checks pass. The repository quality-gate script checks all
  15 operator source/configuration files with zero errors and zero warnings.
- Kernel refactoring preserves the two-pass reduction, buffer allocations and
  synchronization points. ACL smoke ownership now releases resources on errors.
- CANN compilation, ACL device smoke and performance measurements were not
  rerun on this Windows host because the CANN toolkit and NPU are unavailable.
  Earlier device/performance numbers above remain historical measurements.
