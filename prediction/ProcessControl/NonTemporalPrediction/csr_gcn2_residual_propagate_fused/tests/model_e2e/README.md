<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Cora GCNII Reproduction

`benchmark_gcn2_cora_e2e.py` trains or loads the 100-epoch maintained 64-layer
GCNII checkpoint, measures the isolated stage, and replaces all 64 calls for
complete-model E2E. `profile_gcn2_hotspot.py` produces the Level1 complete-model
NPU profile. The checkpoint is generated locally and is not source-controlled;
its hash and formal raw results are recorded in `docs/benchmark.md`. The two
formal JSON files in this directory preserve the component/E2E measurements
and profiler attribution used by the report.
