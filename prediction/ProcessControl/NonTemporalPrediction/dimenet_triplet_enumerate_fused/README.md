<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# DimeNetTripletEnumerateFused

`DimeNetTripletEnumerateFused` constructs the directional `k -> j -> i`
triplets consumed by PyG DimeNet and DimeNet++. Given an edge list in
destination CSR order, it enumerates every incoming edge `k -> j` for each
edge `j -> i`, removes immediate backtracking (`k == i`), and returns the five
node/edge index arrays used by the spherical-basis and interaction blocks.

This is a topology operator, not another message softmax or weighted scatter.
It addresses the maintained PyG call path:

```text
DimeNet.forward -> triplets(edge_index, num_nodes)
```

The stock path requires `torch-sparse`, which is absent in the validation
environment. The performance denominator is therefore a correct resident NPU
tensor implementation, not the unavailable PyG path.

## Contract

```text
row_ptr       int32 [N + 1]       destination CSR
source_index  int32 [E]           source node for every CSR edge
idx_i         int32 [capacity]    triplet target node
idx_j         int32 [capacity]    triplet middle node
idx_k         int32 [capacity]    triplet source node
idx_kj        int32 [capacity]    edge id of k -> j
idx_ji        int32 [capacity]    edge id of j -> i
counts        int32 [2]           {written_triplets, overflow_flag}
```

The integration guard requires validated CSR and
`capacity >= E * max_degree`. Unsupported layouts, dtypes, malformed CSR,
unproven capacity, or overflow use the resident fallback. Outputs and
workspace are independently allocated and recorded on the current NPU stream.

## Value Result

On Ascend 910B3, CANN 8.1.RC1.alpha001, PyTorch/torch_npu 2.5.0/2.5.1,
and PyG 2.9.0, real QM9 batches in a three-block DimeNet++ forward showed:

| QM9 graphs | Triplet stage speedup | Full-model reduction |
|---:|---:|---:|
| 16 | 8.59x | 16.51% |
| 32 | 9.07x | 16.01% |
| 64 | 9.86x | 19.70% |

Every triplet index and every model output matched exactly. The DimeNet++
weights are deterministic initialized weights; this evidence does not claim a
trained or official checkpoint metric.

## Build And Test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
ctest --test-dir build --output-on-failure
python -m pytest tests/test_reference.py tests/test_dispatch.py -q
```

See `docs/benchmark.md` and the JSON artifacts under `tests/` for the exact
baseline, runtime, stream, stress, and model evidence.
