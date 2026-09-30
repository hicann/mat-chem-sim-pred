# Algorithm

Input:

- `order [E]`: int32 permutation of edge IDs in descending edge-score order.
- `edge_index [2,E]`: int32 source/target node IDs.
- `score [E]`: FP32 score in PyG EdgePooling semantics (softmax score plus
  `add_to_edge_score`).
- `nodes N`.

The kernel initializes every cluster to unmatched. It scans `order`
sequentially. An edge is selected only when both endpoints are still unmatched;
both endpoints receive the next selected-cluster ID. After all ranks are
consumed, every unmatched node becomes a singleton cluster in node order.

Outputs are:

- `cluster [N]`
- `selected_edge [N]`, valid for the first `selected_count`
- `cluster_score [N]`, selected edge score or 1.0 for singleton
- `counts[2] = [selected_count, cluster_count]`

The single-block implementation preserves PyG's exact sequential dependency.
Parallel matching is intentionally avoided because endpoint conflicts make the
result order-dependent.

This matches the sequential matching and singleton-assignment portion of PyG
`EdgePooling._merge_edges`. The integration then performs native feature
`scatter(sum)`, score multiplication, `coalesce(cluster[edge_index])`, and
batch remapping. Those operations remain inside complete-model timings.
