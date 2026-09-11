## Summary

Add `CsrTagcnBasisK3Fused`, a forward-only AscendC operator that generates the
four bases used by maintained PyG `TAGConv(K=3)`. Three ordered AIV kernels fuse
the three CSR propagations in one ACL call while all trained Linear transforms
remain on Cube.

## Value

On a 100-epoch two-layer TAGCN Cora checkpoint, one/two/four graph copies improve
from `2.324/4.700/9.788 ms` to `1.327/1.994/3.697 ms`, reducing complete-model
latency by `42.91%/57.57%/62.23%`. The isolated basis stage improves by
`2.16x/2.87x/3.64x`; a Level1 complete-model NPU profile attributes
`87.72%/90.82%/91.60%` of kernel time to that stage. Accuracy remains 79.60%,
predictions agree by at least 99.999994%, and maximum model error is 5.72e-6.

## Deliverables

- Three-stage AscendC kernel, ACL launcher, and OpDef
- NumPy reference, positive/negative tests, and ACL smoke
- Three-scale hotspot, component, and E2E raw benchmarks
- API, algorithm, benchmark, test, and reproduction documentation

## Validation

- Release build and OpDef link: passed on CANN 8.1.RC1.alpha001
- NumPy tests: 7 passed
- CTest and ACL smoke: 1/1 passed, direct ACL output validated
- Python compile and Ruff: passed
