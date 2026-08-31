<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Benchmark

Real QM9 batches were run through a three-block DimeNet++ forward on Ascend
910B3. The denominator is the correct resident NPU triplet-angle path, with the
same topology and deterministic initialized model weights.

| Graphs | Triplets | Resident (ms) | Custom (ms) | Reduction | Model max error |
|---:|---:|---:|---:|---:|---:|
| 16 | 59038 | 86.694 | 18.810 | 78.30% | 1.72e-05 |
| 32 | 87590 | 128.489 | 18.224 | 85.82% | 6.10e-05 |
| 64 | 173204 | 249.289 | 23.305 | 90.65% | 7.63e-06 |

The machine-readable artifact is
`tests/model_e2e/dimenet_qm9_angle_final_20260818.json`. No trained-checkpoint
accuracy claim is made. TorchAir was not validated for this ACLNN path.
