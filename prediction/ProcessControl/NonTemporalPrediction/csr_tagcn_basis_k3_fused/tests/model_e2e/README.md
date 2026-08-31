<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Cora TAGCN Reproduction

`benchmark_component.py` measures three-hop basis generation.
`benchmark_tagcn_cora_e2e.py` trains or loads the maintained two-layer PyG TAGCN
checkpoint and replaces only basis generation. `profile_tagcn_hotspot.py`
produces the Level1 complete-model NPU kernel profile. The checkpoint is
generated locally and is not source-controlled; its hash and formal raw results
are recorded in `docs/benchmark.md`.
