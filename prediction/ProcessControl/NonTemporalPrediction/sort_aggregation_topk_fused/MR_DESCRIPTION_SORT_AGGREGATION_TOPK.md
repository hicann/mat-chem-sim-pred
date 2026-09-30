<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# feature: add fused SortAggregation top-k operator

## Value

Add `SortAggregationTopkFused` for maintained PyG DGCNN graph-classification
inference. The operator replaces dense-batch materialization, full sort,
gather, padding, and flattening with one graph-parallel Ascend C launch while
preserving PyG's stable last-channel ordering and zero-padding semantics.

On Ascend910B3, four MUTAG model sizes show `14.64x-20.29x` replacement-stage
speedup and `58.47%-80.28%` full-model latency reduction against the strongest
correct resident baseline. Prediction agreement is 100%, task accuracy is
unchanged, and maximum model error is `2.87e-6`.

## Deliverables and validation

- Ascend C kernel, Host API, OpDef, and standalone CMake build.
- NumPy reference, five correctness/contract tests, and real ACL smoke.
- Maintained PyG model integration, MUTAG checkpoint, four-shape model E2E,
  dispatch benchmark, and raw JSON.
- Algorithm, API, performance, test, and reproduction documentation.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_sort_aggregation_topk_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_sort_aggregation_topk_fused 0
```

Use PyG fallback for fewer than four graphs, non-contiguous graph membership,
non-FP32 features, non-finite sort keys, or dimensions outside the documented
limits.
