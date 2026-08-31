<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# PPFPointPairFeaturesFused

This operator implements the PyG `point_pair_features` geometry used by
PPFConv/PPFNet: distance plus the three normal/offset angles for every directed
point edge. It removes the CPU `scatter_reduce` fallback from the maintained
point-cloud path while leaving graph construction and feature aggregation under
the model's control.

## Contract

`position` and `normal` are contiguous, zero-offset FP32 `[N, 3]` tensors;
`source_index` and `target_index` are contiguous, zero-offset int32 `[E]`
tensors on the current NPU. The output is an independently allocated FP32
`[E, 4]` tensor in the order `(distance, angle(normal_target, offset),
angle(normal_source, offset), angle(normal_target, normal_source))`. Empty or
unsupported inputs use the resident fallback. Workspace is zero bytes and all
inputs/output are recorded on the current stream.

## Evidence

On Ascend 910B3 with CANN 8.1 RC1 alpha001, PyTorch 2.5.0/torch_npu 2.5.1,
and PyG 2.9, a three-layer PyG PPFConv classifier on real ModelNet10 test
clouds measured against the fastest correct resident path (resident
aggregation shared):

| clouds | resident ms | custom ms | reduction | model max error | prediction agreement |
|---:|---:|---:|---:|---:|---:|
| 4 | 68.470 | 2.941 | 95.70% | 2.98e-08 | 1.0 |
| 8 | 135.581 | 2.909 | 97.85% | 2.98e-08 | 1.0 |
| 16 | 264.694 | 3.915 | 98.52% | 4.47e-08 | 1.0 |

The exact JSON is `tests/model_e2e/ppfnet_modelnet10_e2e_final_20260818.json`.
Weights are deterministic initialized weights; no trained or official
checkpoint quality claim is made. Native PyG emits a CPU fallback warning for
`scatter_reduce`; resident is the stronger correctness denominator.

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
ctest --test-dir build --output-on-failure
python -m pytest tests/test_reference.py tests/test_dispatch.py -q
```

TorchAir was not used for this geometry-only benchmark; the model-level audit
therefore reports the resident eager path explicitly rather than implying
graph capture support.
