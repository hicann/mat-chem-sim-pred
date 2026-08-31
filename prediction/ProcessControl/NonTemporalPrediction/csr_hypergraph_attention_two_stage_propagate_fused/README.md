<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrHypergraphAttentionTwoStagePropagateFused

AscendC implementation of the complete two-stage propagation in PyG
`HypergraphConv(use_attention=True, attention_mode="node")`. It normalizes node
attention within each hyperedge, performs node-to-hyperedge propagation, reuses
the same normalized attention, and performs hyperedge-to-node propagation.

This does not duplicate `CsrHypergraphTwoStagePropagateFused`: that existing op
implements the non-attention path and cannot represent incidence logits or
reuse normalized attention across both stages.

On a Cora neighborhood hypergraph with `H=2,C=8`, the no-grad full model path is
1.314 ms versus the strongest correct resident path at 2.879 ms: 2.19x (54.4%
lower latency), with `2.98e-08` max error. Against PyG eager it is 3.72x, but
that weaker number is not used for acceptance. The machine-readable record is
[`tests/model_e2e/model_e2e_hypergraph_c8_formal_run.json`](tests/model_e2e/model_e2e_hypergraph_c8_formal_run.json).

Limits are FP32, FP16, or BF16 with FP32 accumulation, `H<=4`, `C<=32`, at most
512 nodes per hyperedge and 512 hyperedges per node. The supplied
`torch.library` binding supports training through a numerically checked autograd
recomputation; it does not claim a dedicated AscendC backward kernel. Invalid
or unsupported cases fall back to PyG.

```bash
cmake -S . -B build -DSOC_VERSION=Ascend910B1 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest tests -q
```
