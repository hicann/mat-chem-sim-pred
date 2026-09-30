# Algorithm

PyG `GENConv` with softmax aggregation forms one message per source edge:

```text
m[e, c] = ReLU(features[source_index[e], c]) + 1e-7
```

For destination row `r` and feature channel `c`, the result is:

```text
w[e, c] = exp(t * m[e, c] - max_k(t * m[k, c]))
output[r, c] = sum_e(w[e, c] * m[e, c]) / sum_e(w[e, c])
```

The kernel assigns CSR rows across AIV cores. It scans each row twice. The
first scan gathers messages and computes the per-channel stable maximum. The
second scan computes denominators and weighted numerators. Only five
channel-sized UB buffers are used; no `[N, max_degree, C]` message or softmax
tensor is materialized in global memory.

Malformed row ranges, out-of-range source indices, and empty rows are guarded
on device and produce a zero row. Host validation rejects unrepresentable
dimensions and non-finite temperatures.
