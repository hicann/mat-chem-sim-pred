# feature: add full CSR APPNP propagation operator

## Value

Add `CsrAppnpPropagateFused` for the complete fixed-iteration propagation phase
of maintained `torch_geometric.nn.APPNP`. One public operator owns the
ping-pong workspace and the ten ordered vector-kernel launches, removing the
Python/framework loop and intermediate tensor materialization. The model MLP
remains native NPU code.

On Ascend910B3, five full Cora shapes reduce end-to-end model latency by
`18.05%-45.59%` against the faster of official PyG APPNP and an equivalent
resident torch_npu rewrite. Default Cora latency is `2.932 -> 1.595 ms`.
Checkpoint test accuracy remains `0.8270`, prediction agreement is effectively
100%, and maximum model error is `5.72e-6`.

## Deliverables and validation

- Ascend C vector kernel, Host API, OpDef, standalone CMake, and ping-pong workspace.
- NumPy reference, four correctness/contract tests, and direct ACL smoke test.
- Official PyG APPNP checkpoint, five-shape model E2E, dispatch scan, and raw JSON.
- Algorithm, API, performance, test, fallback, and reproduction documentation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_csr_appnp_propagate_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_csr_appnp_propagate_fused 0
```

Framework fallback remains required for training-time dropout, non-FP32 data,
unsupported dimensions/layouts, or graphs below the validated dispatch floor.
