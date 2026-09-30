# PROTEINS model value gate

`train_edgepool_proteins.py` creates a fixed-seed local checkpoint for a
`GraphConv -> EdgePooling -> GraphConv -> global mean -> Linear` classifier on
the real PyG PROTEINS dataset. The split is a deterministic random 80/20 split,
not the class-skewed contiguous dataset suffix. The checked-in checkpoint is a
local reproduction artifact, not an official upstream checkpoint.

`benchmark_edgepool_proteins_cpu_npu_e2e.py` compares the complete trained
model on one node202 host: CPU PyG with a thread sweep, maintained PyG NPU,
optimized equivalent resident NPU, custom resident NPU, and custom host-to-host
including dynamic graph H2D and logits D2H. It also verifies exact cluster,
pooled feature, coalesced edge, batch, logit, prediction, and task behavior.

The node202 artifact `edgepool_proteins_cpu_npu_e2e_node202_20260902.json`
contains B16/B32/B128 results. Custom reduces the strongest resident baseline
by `13.34%/21.39%/29.75%` and is `12.77x/25.50x/41.05x` faster than CPU in the
host-to-host boundary. The checked-in full-test script has not yet produced a
223-graph NPU result.
