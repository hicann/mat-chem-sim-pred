<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Algorithm

For graph `g`, let its contiguous node rows be `X_g[n,c]`. PyG
`SortAggregation(k)` orders rows by the final channel in descending order,
preserves input order for equal keys, retains the first `k` rows, pads missing
rows with zero, and returns `[graphs, k * channels]`.

```text
order_g = stable_argsort(X_g[:, channels - 1], descending=True)
Y[g, j, :] = X_g[order_g[j], :]  when j < min(k, nodes_g)
Y[g, j, :] = 0                    otherwise
```

The Ascend C kernel assigns graphs across at most 40 AI Vector cores. Each
core maintains only `k` scores and source indices in UB using stable insertion,
then copies selected complete feature rows directly from GM to the flattened
output. This avoids PyG's dense-batch materialization, fill-mask update,
full-width sort, gather, and padding kernels.

The custom integration requires contiguous graph rows represented by INT32
`graph_ptr`, finite FP32 sort keys, `1 <= channels <= 1024`, and
`1 <= k <= 128`. The validated dispatch selects the custom path for at least
four graphs; unsupported layouts, dtypes, or dimensions use PyG.
