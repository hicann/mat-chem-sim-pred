<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Cora ChebNet Reproduction

`benchmark_component.py` measures K=3 basis generation. The E2E script trains
or loads a 100-epoch two-layer PyG ChebNet checkpoint and replaces only basis
generation. `profile_chebnet_hotspot.py` produces a Level1 complete-model NPU
kernel profile and attributes only the kernel types emitted by the exact native
basis operations. The checkpoint is generated locally and is not
source-controlled; its SHA-256 and formal-library raw results are recorded in
`docs/benchmark.md`.
