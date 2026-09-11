# Algorithm

For neighbor projection `L[N,C]`, root projection `R[N,C]`, and CSR row `i`:

```text
Y[i,c] = ReLU(R[i,c] + mean(L[column_index[e],c], e in row i))
```

An empty row contributes a zero neighbor mean. For official PyG `SAGEConv`,
`L = Linear_left(X, bias=False)` and
`R = Linear_root(X) + bias_left`. This equals the original
`Linear_left(mean(X_neighbors)) + Linear_root(X)` for every non-empty row and
also preserves the left bias for an empty row.

The kernel assigns rows across at most 40 cores. A channel-aligned UB buffer
accumulates projected neighbors with vector `Add`, scales once by inverse
degree, adds the root row, applies vector `Maxs`, and copies the logical output
width to GM. Supported inputs are INT32 CSR and FP32 features with
`1 <= channels <= 1024`.

The custom path is validated for `nodes >= 64`; smaller, unsupported-dtype, or
out-of-contract shapes use the framework implementation.
