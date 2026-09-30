# EdgepoolSortedRankContractFused

AscendC implementation of the topology-changing stage in PyG `EdgePooling`:
given edge scores already sorted in descending order, it performs the exact
greedy endpoint matching, assigns contracted and singleton cluster IDs, emits
selected edge IDs and per-cluster scores, and returns dynamic selected/cluster
counts.

This operator intentionally does not duplicate the generic segment-softmax
aggregation kernels already delivered for graph attention. Its distinct value is
removing PyG EdgePooling's Python loop over every ranked edge and the repeated
CPU tensor sync on this runtime.

The fixed-seed local PROTEINS checkpoint reaches 74.89% test accuracy. On
deterministic B16/B32/B128 test batches, cluster, pooled feature, coalesced
edge, batch, and resident-NPU logits are exact. Full-model latency improves
from the strongest non-custom `5.371/7.178/14.099 ms` to
`4.654/5.643/9.904 ms`, reductions of `13.34%/21.39%/29.75%`. Dynamic-graph
host-to-host custom latency is `4.989/5.943/10.082 ms`, or
`12.77x/25.50x/41.05x` faster than same-host CPU PyG.

The current covered path is FP32 inference with int32 descending permutation,
int32 edge indices, and prevalidated finite edge scores. The first release does
not claim a custom backward or mixed-precision contraction. The compatibility
wrapper copies dynamic counts to the host before returning valid views. For GE,
the static API instead returns fixed-capacity tensors plus a device count; its
custom OPP passes FP32 TorchAir `fullgraph=True` with exact outputs.

The operator boundary ends at cluster/selected-edge/cluster-score/counts.
Feature reduction, `coalesce` topology reconnection, and batch remapping remain
native and are included in complete-model timings. The checkpoint is local,
not official. Full 223-graph task regression now passes for CPU, resident NPU,
and custom NPU (`74.8879%` accuracy; resident/custom predictions exactly
agree). Timing evidence uses the deterministic B16/B32/B128 serving shapes;
the full-task regression artifact is
`tests/model_e2e/edgepool_full_223_task_regression_20260902.json`.
