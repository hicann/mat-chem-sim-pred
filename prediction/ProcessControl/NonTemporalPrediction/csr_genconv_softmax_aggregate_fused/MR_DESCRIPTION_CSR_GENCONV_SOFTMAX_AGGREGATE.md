## Summary

Add `CsrGenConvSoftmaxAggregateFused`, an AscendC inference operator for the
message and `SoftmaxAggregation` portion of PyG `GENConv`. The kernel fuses CSR
source gather, `ReLU+eps`, stable per-feature temperature softmax, and weighted
neighbor reduction without materializing padded edge messages.

## Model Value

The operator targets DeeperGCN and other maintained PyG models using GENConv.
On the tested torch_npu stack, exact PyG `scatter_reduce` falls back to CPU.
Reported speedups use a power-of-two degree-bucketed all-NPU semantic baseline
rather than that fallback or the historical weak global padded baseline.

With an 80-epoch, eight-layer DeeperGCN Cora checkpoint, complete forward
latency for one/two/four disjoint graph copies changes as follows:

| Copies | Fastest CPU | Strong NPU | Custom resident | Custom H2H | NPU reduction |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 497.352 ms (1T) | 20.593 ms | 4.570 ms | 5.924 ms | 77.81% |
| 2 | 573.072 ms (16T) | 19.695 ms | 4.565 ms | 7.095 ms | 76.82% |
| 4 | 698.806 ms (16T) | 18.452 ms | 7.378 ms | 10.595 ms | 60.02% |

The global padded baseline expands source slots by `34.50x`; the accepted
degree-bucket baseline expands them by only `1.309x`. Maximum complete-model
error is `3.34e-6`. CPU and custom Cora test accuracy are both `70.10%`, with
`100%` prediction agreement. The custom H2H boundary includes feature H2D and
logit D2H, while graph topology and weights remain resident. The locally
trained fixed-seed checkpoint is not official.

## Deliverables

- AscendC kernel, ACL host launcher, and framework OpDef
- NumPy reference and positive/negative tests
- ACL direct smoke and CTest integration
- Three-scale component and complete-model benchmarks with raw JSON
- Same-host CPU thread sweep and resident/host-to-host NPU value gate
- Algorithm, API, benchmark, test, and reproduction documentation

## Limitations and Fallback

The custom path is FP32, contiguous, forward-only, and supports channels up to
1024. It handles the `edge_attr=None` GENConv message form. Framework dispatch
must use the native implementation for autograd, non-contiguous inputs unless
materialized, unsupported types or shapes, and edge-attribute message forms.

## Validation

- Release build and OpDef link: passed on Ascend 910B3
- NumPy reference tests: 7 passed
- CTest and direct ACL smoke: passed
- Python compile and Ruff: passed
