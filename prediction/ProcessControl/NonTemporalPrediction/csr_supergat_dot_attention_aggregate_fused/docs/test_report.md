# Test report

- NumPy reference unit tests: stable row softmax, manual result, training,
  non-contiguous, and degree-limit fallback.
- Clean Ascend 910B1 configure/build: passed after host alias hardening.
- Component clean-runtime correctness: max absolute error `1.07e-06`.
- Same-weight PyG model correctness: max absolute error `7.45e-08`.
- Two concurrent streams with separate buffers: passed, error `4.77e-07`.
- Output alias, undersized workspace, and degree 513: all rejected.
- 2026-08-14 hardening: ACLNN CTest, FP32/FP16/BF16 direct checks, and
  `torch.library` lifecycle/export/autograd checks passed.
- 2026-08-16 custom OPP: FP32 TorchAir `fullgraph=True` passed on Ascend910B3;
  output shape `[32,2,8]`, 128 edges, max error `2.38e-7`.

The runtime owns no input or output memory. Buffers remain caller-owned until
the supplied stream completes.
