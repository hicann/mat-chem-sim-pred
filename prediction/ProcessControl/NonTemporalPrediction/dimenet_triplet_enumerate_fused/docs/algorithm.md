<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

The CSR rows are destination nodes. For each edge `j -> i`, the kernel reads
row `j` to enumerate every incoming edge `k -> j`. It emits a triplet unless
`k == i`, matching PyG's immediate-backtracking mask. Iteration follows CSR
edge order and then the incoming-edge order, so the five arrays match PyG's
`SparseTensor` construction order when the same CSR ordering is used.

The first implementation uses one AI Vector core deliberately: output order is
deterministic and no global atomic counter is required. The measured workload
remains 8.59x to 9.86x faster than the strongest correct resident tensor
construction for the validated QM9 shapes. Multi-core prefix-sum enumeration
is a possible later optimization, not required for the current value gate.

The output shape is fixed-capacity. `counts[0]` reports the valid prefix and
`counts[1]` reports malformed input or capacity overflow. The dispatch guard
proves `capacity >= E * max_degree`; callers must reject a non-zero overflow
flag rather than consume partial topology.
