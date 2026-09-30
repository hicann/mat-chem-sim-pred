# Add EdgepoolSortedRankContractFused

## What

Add an AscendC/ACLNN operator for the unique topology-changing contraction in
PyG `EdgePooling`: descending-rank greedy endpoint matching, cluster IDs,
selected-edge IDs, per-cluster scores, and dynamic counts.

## Why

Maintained PyG `_merge_edges` synchronizes ranked edges to Python and loops
over every edge on CPU. On a fixed-seed locally trained PROTEINS checkpoint,
against the faster optimized resident implementation, B16/B32/B128 full-model
medians improve from `5.371/7.178/14.099 ms` to `4.654/5.643/9.904 ms`
(`13.34%/21.39%/29.75%`). Dynamic-graph host-to-host custom is
`12.77x/25.50x/41.05x` faster than same-host CPU PyG. Cluster, pooled feature,
coalesced edge, batch, resident logits, and predictions are exact.

## Risk controls

The ACLNN host validates shapes, pointers, aliases, workspace, and stream. The
guarded dispatcher requires a prevalidated int32 permutation and falls back for
unsupported dtype/layout/shape. A NumPy oracle covers self-loops, singletons,
bad permutations, and out-of-range indices. Clean build, ACLNN CTest, binding
test, 500-call lifecycle stress, dual-current-stream execution, and exact model
A/B pass. The FP32 GE API returns fixed-capacity outputs plus device counts and
passes TorchAir `fullgraph=True`; the Python compatibility API still performs a
host count copy and slicing. No custom backward or mixed precision is claimed.

The checkpoint is a local reproduction artifact (74.89% test accuracy), not an
official upstream checkpoint. Full 223-graph CPU/resident/custom task
regression now passes with `74.8879%` accuracy for all three paths and exact
resident/custom predictions; the machine-readable result is
`tests/model_e2e/edgepool_full_223_task_regression_20260902.json`. The timing
benchmark remains on deterministic B16/B32/B128 serving shapes, so this is
submission-ready with that documented timing boundary.
