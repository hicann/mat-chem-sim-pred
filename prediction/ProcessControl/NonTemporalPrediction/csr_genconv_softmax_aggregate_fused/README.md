# CsrGenConvSoftmaxAggregateFused

`CsrGenConvSoftmaxAggregateFused` accelerates the message and aggregation
portion of PyG `GENConv` with `SoftmaxAggregation`. For every destination node
and feature channel, it fuses source gather, `ReLU(x_j) + 1e-7`, stable
temperature softmax, and the weighted neighbor reduction.

The operator is intended for inference in DeeperGCN-style models when
`edge_attr` is absent or has already been folded into source messages. The
forward-only custom path is not used for training.

## Contract

```text
row_ptr       int32   [N + 1]
source_index  int32   [E]
features      float32 [N, C], contiguous
temperature   finite scalar attribute
output        float32 [N, C]
```

The custom path supports `1 <= C <= 1024`, positive `N` and `E` representable
by INT32, non-decreasing CSR row pointers spanning `[0, E]`, and source indices
in `[0, N)`. Empty rows produce zeros. Callers must use the native path for
unsupported dtypes, layouts, shapes, edge-message variants, or autograd.

## Build and Test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
python -m pytest tests/test_csr_genconv_softmax_aggregate_fused.py -q
ctest --test-dir build --output-on-failure
ASCEND_RT_VISIBLE_DEVICES=6 \
  ./build/examples/test_aclnn_csr_genconv_softmax_aggregate_fused 0
```

## Measured Result

On Ascend 910B3, an 80-epoch, eight-layer PyG DeeperGCN checkpoint on Cora was
benchmarked at one, two, and four disjoint graph copies. A new same-host value
gate replaced the old global maximum-degree padded baseline, which inflated
the edge-message tensor by `34.50x`, with a power-of-two degree-bucketed
all-NPU baseline whose padding ratio is `1.309x`.

Against that strongest correct resident NPU baseline, complete model latency
changed from `20.593/19.695/18.452 ms` to `4.570/4.565/7.378 ms`, a
`77.81%/76.82%/60.02%` reduction. The custom host-to-host path, including
feature H2D and logits D2H while keeping the static graph and weights resident,
measured `5.924/7.095/10.595 ms`. The fastest same-host official PyG CPU
forwards in a `1/4/8/16` thread sweep measured
`497.352/573.072/698.806 ms`. Test accuracy remained `70.10%`, prediction
agreement was `100%`, and maximum model error was `3.34e-6`.

The old `96.85%-98.72%` reduction is retained only as historical evidence
against a weak globally padded baseline and must not be used as the value
claim. The checkpoint is fixed-seed and locally trained, not official.

See `docs/benchmark.md` for the evidence boundary and raw artifact mapping.
