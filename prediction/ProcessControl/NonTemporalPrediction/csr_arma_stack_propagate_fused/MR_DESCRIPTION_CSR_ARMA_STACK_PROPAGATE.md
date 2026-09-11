## Summary

Add `CsrArmaStackPropagateFused`, a forward-only AscendC operator for the
weighted multi-stack propagation, root/bias merge, and optional activation in
maintained PyG `ARMAConv`. Learned projections remain native Cube operations.

## Value

On a 400-epoch Cora checkpoint, one/two/four graph copies improve from
`3.137/5.431/10.049 ms` to `2.039/3.536/6.537 ms`, reducing complete-model
latency by `35.01%/34.90%/34.95%`. The source stage improves by
`1.41x/1.46x/1.49x`; conservative Level1 profiling attributes
`76.95%/81.85%/85.40%` of kernel time to its four repeated calls while
excluding all ReLU kernels. Accuracy stays 79.50%, predictions agree by at
least 99.999994%, and model error is at most 1.53e-5.

Native `ARMAConv` backward hits an `aclnnScatterAdd` error in the tested
torch_npu environment, so the checkpoint was trained with an equivalent
normalized dense adjacency matmul. Re-loading the weights into official PyG
`ARMAConv` gives zero error against the staged model forward.

## Deliverables

- AscendC kernel, ACL launcher, and OpDef with required ReLU attribute
- NumPy reference, positive/negative tests, and ACL device smoke
- Three-scale component, complete-model, and profiler evidence
- API, algorithm, benchmark, test, and reproduction documentation

## Validation

- Release build, AIV kernel, ACL launcher, and OpDef link: passed
- NumPy reference suite: 8 passed
- CTest and ACL device smoke on Ascend 910B3: passed
- Python compile and Ruff: passed
- Three-scale component, E2E, accuracy, and Level1 profiler runs: passed
