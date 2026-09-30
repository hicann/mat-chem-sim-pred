# API reference

`aclnnEdgepoolSortedRankContractFusedGetWorkspaceSize(N,E)` currently returns
one workspace byte for a uniform caller-owned allocation contract. The kernel
does not otherwise consume workspace.

`aclnnEdgepoolSortedRankContractFused(order, edge_index, score, cluster,
selected_edge, cluster_score, counts, N, E, workspace, workspace_size, stream)`
launches asynchronously on `stream`.

Inputs must be contiguous int32 `order [E]`, int32 `edge_index [2,E]`, and
FP32 `score [E]`. All six outputs must be distinct and non-aliasing with all
inputs: int32 `cluster [N]`, int32 `selected_edge [N]`, FP32
`cluster_score [N]`, and int32 `counts [2]`. The caller must validate that
`order` is a permutation, endpoint IDs are in `[0,N)`, and scores are finite.
Unsupported shape/layout/alias cases must use the maintained PyG fallback.
