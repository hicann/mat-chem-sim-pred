<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->
# SortAggregationTopkFused

`SortAggregationTopkFused` implements the maintained
`torch_geometric.nn.aggr.SortAggregation` path used by DGCNN graph
classifiers. For every graph it stably sorts nodes by the last feature,
keeps the first `k` complete rows, zero-pads short graphs, and flattens the
result for the classifier.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
python -m pytest tests/test_sort_aggregation_topk_fused.py -q
LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH" \
  ./build/examples/test_aclnn_sort_aggregation_topk_fused 0
```

On Ascend910B3/CANN 8.1.RC1.alpha001, four real MUTAG DGCNN batch sizes
show `14.64x-20.29x` aggregation speedup and `58.47%-80.28%` full-model
latency reduction against the strongest correct resident baseline. Prediction
agreement is 100%, task accuracy is unchanged, and maximum model error is
`2.87e-6`.
