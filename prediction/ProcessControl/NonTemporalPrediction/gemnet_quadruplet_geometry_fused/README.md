<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# GemNetQuadrupletGeometryFused

This is the geometry half of the official GemNet quadruplet path. It computes
the cached `angle_cab`, `angle_abd`, and projected torsion `theta_cabd` arrays
from the official GemNet interaction/edge topology. It is intentionally
delivered separately from `GemNetQuadrupletEnumerateFused`: topology creation
and geometry have different contracts, fallback behavior, and model audit
results.

## Contract and ownership

All inputs are contiguous, zero-offset NPU tensors: FP32 `position [N,3]`,
int32 edge/interactions and cache indices. Output tensors are independently
allocated FP32 arrays with lengths `intermediate_ca`, `intermediate_db`, and
`quadruplets`. The host API has zero workspace, validates positive bounded
sizes, launches on the caller's current stream, and the binding records every
input/output on that stream. Unsupported dtypes/layouts/empty shapes use the
resident official GemNet geometry path.

## Model evidence

Against the official GemNet source's resident `calculate_angles` implementation
on real QM9 batches, with the same fixed random initialization:

| graphs | native ms | custom ms | speedup | angle max error | model max error |
|---:|---:|---:|---:|---:|---:|
| 1 | 224.944 | 34.121 | 6.59x | 5.01e-06 | 7.63e-06 |
| 2 | 375.105 | 44.922 | 8.35x | 5.01e-06 | 7.63e-06 |

The exact artifact is `tests/model_e2e/gemnet_geometry_e2e_independent_20260818.json`.
The model
uses deterministic initialized weights and an empty scale file; this is a
structural/E2E performance claim, not an official checkpoint result. The
official source still requires a small `torch_scatter` compatibility shim in
this validation environment. TorchAir was not runnable for this source path,
so the resident eager implementation is the strongest correct denominator.

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
ctest --test-dir build --output-on-failure
python tests/smoke_geometry_npu.py --build-dir build
```
