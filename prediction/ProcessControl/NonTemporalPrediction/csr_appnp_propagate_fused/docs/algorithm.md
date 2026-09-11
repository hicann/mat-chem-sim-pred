# Algorithm

For normalized CSR adjacency `A`, initial logits `H`, teleport probability
`alpha`, and `K` iterations, the operator computes:

```text
X(0) = H
X(k+1) = (1 - alpha) * A * X(k) + alpha * H
```

The Host API allocates two workspace tensors and launches one vectorized CSR
step per iteration on the same stream. Every row is owned by one AI Vector Core;
feature vectors are accumulated in UB. The final iteration writes directly to
the caller output. This preserves PyG APPNP inference semantics for dropout zero
and precomputed `gcn_norm` weights.

The output must not alias `initial` because every iteration reads the original
teleport state. Empty CSR rows are valid and produce `alpha * initial[row]`.
