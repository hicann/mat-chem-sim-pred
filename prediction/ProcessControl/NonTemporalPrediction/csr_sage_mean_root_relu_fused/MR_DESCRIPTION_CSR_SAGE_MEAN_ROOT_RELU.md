# feature: add CSR GraphSAGE mean/root/ReLU fused operator

## Value

Add `CsrSageMeanRootReluFused` for the neighbor mean, root-branch addition,
and ReLU in maintained PyG `SAGEConv`. Both learned Linear projections remain
native NPU operations. Moving the bias-free neighbor projection before the
mean and folding its bias into the root branch is algebraically exact,
including empty neighbor rows.

On Ascend910B3, five full Cora model shapes show `1.61x-2.74x` replacement
stage speedup and `20.21%-49.60%` end-to-end latency reduction against the
faster of the official PyG path and an equivalent resident torch_npu rewrite.
Checkpoint test accuracy remains `0.7120`, prediction agreement is effectively
100%, and maximum model error is `4.77e-7`.

## Deliverables and validation

- Ascend C kernel, Host API, OpDef, and standalone CMake build.
- NumPy reference, four correctness/contract tests, and a real ACL smoke test.
- Maintained PyG SAGEConv checkpoint, five-shape model E2E, and raw JSON.
- Algorithm, API, performance, test, and reproduction documentation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_csr_sage_mean_root_relu_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_csr_sage_mean_root_relu_fused 0
```

Use the framework fallback for inputs below the validated dispatch floor,
non-FP32 features, or channel counts outside the documented limit.
