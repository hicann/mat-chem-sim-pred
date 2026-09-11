## Summary

Add `CsrGcn2ResidualPropagateFused`, a forward-only AscendC operator for the
normalized propagation plus initial residual in maintained PyG `GCN2Conv`.
The subsequent trained `addmm` remains on Cube.

## Value

On a 100-epoch, 64-layer Cora GCNII checkpoint, one/two/four graph copies
improve from `22.460/39.360/73.973 ms` to `13.109/17.928/30.334 ms`, reducing
complete-model latency by `41.63%/54.45%/58.99%`. The stage improves by
`1.55x/1.79x/2.11x`; Level1 profiling attributes
`91.21%/94.83%/96.80%` of kernel time to its 64 repeated calls. Accuracy stays
79.20%, predictions agree by at least 99.999994%, and model error is at most
1.91e-6.

## Deliverables

- AscendC kernel, ACL launcher, and OpDef with alpha attribute
- NumPy reference, positive/negative tests, and ACL smoke
- Three-scale component, complete-model, and profiler evidence
- API, algorithm, benchmark, test, and reproduction documentation

## Validation

- Release build, AIV kernel, ACL launcher, and OpDef link: passed
- NumPy reference suite: 7 passed
- CTest and ACL device smoke on Ascend 910B3: passed
- Python compile and Ruff: passed
- Three-scale component, E2E, accuracy, and Level1 profiler runs: passed
