<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Cora GAT Reproduction

`benchmark_gat_cora_e2e.py` uses the complete 13,264-edge CSR, measures both
eligible model layers, and compares custom execution with official PyG and an
exact resident NPU baseline in reversed trial order. `profile_gat_hotspot.py`
profiles the resident NPU complete model conservatively. The checkpoint is not
checked in; its SHA-256 and task accuracy are recorded in `docs/benchmark.md`.

The checked-in raw formal outputs are `gat_cora_formal_20260802.json` and
`gat_cora_hotspot_formal_20260802.json`. The benchmark aborts if any graph row
is truncated or if the required external checkpoint is absent.
