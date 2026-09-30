# CsrAgnnCosineAttentionAggregateFused

`CsrAgnnCosineAttentionAggregateFused` implements the inference attention
stage used twice by maintained PyG `examples/agnn.py`: cosine score, stable CSR
segment softmax, and weighted feature aggregation. The framework keeps the
linear layers, ReLU, and `F.normalize` native and passes both the value tensor
and its normalized view to this operator.

The upstream call point is `torch_geometric.nn.conv.AGNNConv`: `forward()`
normalizes node features and `message()` computes `beta * cosine`, target-wise
`softmax`, and weighted `x_j`. This is not a duplicate of the generic
`CsrAttentionAggregateFused`, which consumes already materialized edge logits,
or of SuperGAT/PointTransformer operators, whose score and value semantics are
different. This operator removes AGNN's endpoint gathers, edge-score tensor,
segment softmax, weighted edge tensor, and scatter reduction as one model-aware
stage.

The forward path supports contiguous INT32 CSR and FP32 `[N,C]` values with
`C<=512`, non-empty valid CSR, finite `beta`, and row size at most 2048. Other
dtypes, layouts, shapes, malformed graphs, autograd, or aliased output must use
the native path. Formal component, complete-model, task, profiler, reference,
and ACL evidence is recorded under `docs/` and `tests/model_e2e/`.

On node202, same-host complete-model validation includes a CPU thread sweep,
official PyG NPU, custom resident NPU, and a static-graph host-to-host path. The
custom host-to-host path is `4.22x-6.76x` faster than the best same-host CPU
configuration, while resident custom NPU lowers latency by `46.99%-56.36%`
against the fastest correct non-custom path. Cora test accuracy remains 81.80%.
