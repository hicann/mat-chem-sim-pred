# Model E2E Reproduction

After building the operator and setting `LD_LIBRARY_PATH` to `build` and
`build/lib`, run:

```bash
python tests/model_e2e/benchmark_pyg_appnp_e2e.py \
  --dataset-root /path/to/Planetoid \
  --checkpoint /path/to/cora_pyg_appnp_20260801.pt \
  --build build \
  --output tests/model_e2e/cora_pyg_appnp_formal_20260801.json \
  --copies 1 4 16 32 64 --warmup 10 --repeat 50

python tests/model_e2e/benchmark_shape_dispatch.py \
  --build build \
  --output tests/model_e2e/csr_appnp_dispatch_20260801.json
```

The checkpoint stores the official PyG model state and validation/test metrics.
The optional `--bootstrap-checkpoint` converts the earlier equivalent MLP state;
all recorded metrics are recomputed through the official full-edge PyG call.
