<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrFilmModulatedMeanFused

Inference-only AscendC operator for PyG `FiLMConv`. It fuses per-edge
feature-wise linear modulation, optional message ReLU, and target-wise mean
aggregation into one kernel.

## Model value

The integration target is the maintained PyTorch Geometric `examples/film.py`
topology at revision `003c3cd8a10520567ceaeda619f0315e30ec2f66`: four
`FiLMConv` layers with 320 hidden channels on the real PPI dataset. The first
three layers use message ReLU; the final 121-channel layer disables it.

On Ascend 910B3, the first replaceable stage is `4.18x/5.82x/6.65x` faster for
two, four, and eight PPI graphs. Replacing all four model stages reduces E2E
latency against official PyG by `60.29%/69.05%/72.31%`:

| PPI graphs | Nodes | Edges | Official PyG | Fused | E2E reduction |
|---:|---:|---:|---:|---:|---:|
| 2 | 3,144 | 62,022 | 9.034 ms | 3.588 ms | 60.29% |
| 4 | 6,288 | 124,044 | 17.055 ms | 5.278 ms | 69.05% |
| 8 | 12,576 | 248,088 | 32.952 ms | 9.123 ms | 72.31% |

The 50-epoch checkpoint reaches 0.8866 test micro-F1. All 121-dimensional
binary predictions match official PyG at every measured size. Maximum logit
error divided by maximum output magnitude is at most `2.56e-7`.

## Supported contract

- `row_ptr`: contiguous `int32`, shape `[N + 1]`.
- `source_index`: contiguous `int32`, shape `[E]`.
- `projected`, `beta`, `gamma`: contiguous `float32`, shape `[N, C]`.
- `1 <= C <= 512`, `E > 0`, maximum CSR row length at most 2,048.
- `apply_relu` is a boolean and inputs are finite.
- inference only; unsupported dtype, shape, layout, training, or fan-in must
  dispatch to native PyG.

The output is contiguous `float32` with shape `[N, C]`. Empty rows return zero,
matching mean aggregation semantics.

## Build and test

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cmake -S . -B build -DSOC_VERSION=Ascend910B3 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j4
python -m pytest -q
ctest --test-dir build --output-on-failure
```

Raw measurements and profiler aggregation are under `tests/model_e2e`.
