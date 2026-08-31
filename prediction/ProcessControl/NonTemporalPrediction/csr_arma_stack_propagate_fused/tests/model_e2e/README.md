<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Cora ARMA Reproduction

`benchmark_arma_cora_e2e.py` loads the fixed 400-epoch checkpoint, measures the
isolated source stage, and replaces all four eligible calls in the complete
two-layer model. `profile_arma_hotspot.py` records conservative Level1 kernel
attribution; ReLU is excluded because one additional ReLU occurs outside the
replaceable stage. The checkpoint is generated locally and is not checked in;
its SHA-256 and task accuracy are recorded in `docs/benchmark.md`.
The two formal JSON files in this directory preserve the component/E2E
measurements and conservative profiler attribution used by the report.
