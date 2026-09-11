# Test Report

- Clean Ascend910B3 Release build: passed.
- NumPy deterministic, random, dispatch, and Host contract tests: 4 passed.
- CTest and direct ACL smoke: passed; output
  `[2.46875,-0.125,3.0625,2.25]` matches the CPU reference.
- Invalid workspace, non-finite alpha, and output/initial alias cases return
  `ACL_ERROR_INVALID_PARAM`.
- Official PyG APPNP checkpoint SHA256:
  `e25b574448c2880d1fe31869465a47baf19753c5127ba8c72e78dd5276025d2c`.
- Cora validation/test accuracy: `0.7980/0.8270`.
- Five model shapes: maximum model error `5.72e-6`, task metric unchanged.

The model benchmark includes the complete MLP and ten APPNP propagation steps,
uses all Cora edges, and compares against both official PyG and resident
torch_npu paths.
