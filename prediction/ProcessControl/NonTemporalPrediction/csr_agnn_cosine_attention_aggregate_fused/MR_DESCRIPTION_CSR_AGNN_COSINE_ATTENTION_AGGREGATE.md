## Summary

Add `CsrAgnnCosineAttentionAggregateFused`, a forward-only AscendC operator
for the cosine score, stable CSR segment softmax, and weighted aggregation used
twice by maintained PyG AGNN. Linear projection, activation, normalization,
and unsupported cases remain on the native path.

## Value

The initial resident-only Cora run improved one/four/sixteen complete graphs
from `3.611/5.552/62.580 ms` to `1.359/2.881/8.861 ms`. The strict node202
same-host value gate is the acceptance result: custom host-to-host latency is
`2.607/6.416/20.774 ms`, versus same-host CPU `12.839/43.404/87.636 ms`, so
the custom path is `4.92x/6.76x/4.22x` faster than CPU. Against the fastest
correct non-custom NPU path, resident custom lowers model latency by
`56.36%/46.99%/48.44%`. Test accuracy remains 81.80%, prediction agreement is
100%, and maximum model error is 4.77e-7.

The stricter node202 same-host gate sweeps CPU PyG over 1/4/8/16 threads and
also measures feature-H2D through logits-D2H with static graph/model state
resident. Best CPU is `12.839/43.404/87.636 ms`; custom host-to-host is
`2.607/6.416/20.774 ms`, or `4.92x/6.76x/4.22x` faster. Against the fastest
correct non-custom NPU path, resident custom lowers model latency by
`56.36%/46.99%/48.44%`.

The exact PyG call point is `AGNNConv.forward/message`. Unlike the generic CSR
attention aggregate, this operator computes AGNN's beta-scaled endpoint cosine
score inside the fused stage; it does not consume precomputed logits. SuperGAT
and PointTransformer use different score/value semantics.

## Validation

Release build, OpDef, NumPy reference tests, negative cases, ACL device smoke,
three-scale component/E2E, task parity, profiler, Python compile, and Ruff all
pass on Ascend 910B3 with CANN 8.1.RC1.alpha001.

Complete-model TorchAir fullgraph is not claimed; the official PyG NPU route
warns that `scatter_reduce` falls back to CPU.
