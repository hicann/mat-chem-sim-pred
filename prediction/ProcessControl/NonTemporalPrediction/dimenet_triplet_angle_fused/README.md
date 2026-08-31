<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# DimeNetTripletAngleFused

This operator computes the DimeNet/DimeNet++ triplet angle
`atan2(||(r_i-r_j) x (r_j-r_k)||, (r_i-r_j) dot (r_j-r_k))` for a resident
triplet topology. It targets the geometry stage after triplet enumeration;
the separate `DimeNetTripletEnumerateFused` operator owns topology creation.

The contract is contiguous, zero-offset FP32 `position [N,3]` and int32
`idx_i/idx_j/idx_k [T]` on NPU. The output is independently allocated FP32
`[T]`; invalid layouts, dtypes, empty triplets, and unsupported shapes use the
resident fallback. Workspace is zero bytes and tensors are recorded on the
current stream. The kernel uses a bounded Cephes-style atan2 approximation;
the model-level error is reported below rather than hidden behind compile-only
evidence.

## Evidence

On Ascend 910B3 with CANN 8.1 RC1 alpha001, PyTorch 2.5.0/torch_npu 2.5.1,
and PyG 2.9, real QM9 batches through a three-block DimeNet++ forward measured
against resident NPU triplet geometry:

| graphs | triplets | resident ms | custom ms | reduction | model max error |
|---:|---:|---:|---:|---:|---:|
| 16 | 59038 | 86.694 | 18.810 | 78.30% | 1.72e-05 |
| 32 | 87590 | 128.489 | 18.224 | 85.82% | 6.10e-05 |
| 64 | 173204 | 249.289 | 23.305 | 90.65% | 7.63e-06 |

The exact JSON is `tests/model_e2e/dimenet_qm9_angle_final_20260818.json`.
Weights are deterministic initialized weights; no trained or official
checkpoint quality claim is made. This is a geometry candidate distinct from
triplet enumeration, not another scatter/softmax aggregate.

## Validation

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j2
ctest --test-dir build --output-on-failure
python -m pytest tests/test_dispatch.py -q
```

The model benchmark uses the resident eager path as the strong correct
denominator. TorchAir was not run for this custom ACLNN geometry path.
