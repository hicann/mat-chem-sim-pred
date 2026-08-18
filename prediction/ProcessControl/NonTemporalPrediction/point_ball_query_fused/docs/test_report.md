<!-- Copyright (c) 2026 Huawei Technologies Co., Ltd. Licensed under the CANN Open Software License Agreement Version 2.0. -->

# Test Report

- Independent Release build: PASS on node202 / Ascend 910B3.
- Standalone host/kernel NPU smoke: PASS; two deterministic neighborhoods are
  exactly `[[0,1],[1,2]]`.
- NumPy reference test: PASS.
- Pretrained PointNet++ model parity: zero index mismatches, zero logit error,
  and 100% top-1 agreement at batches 1, 4, and 8.
- Invalid dimensions, non-positive radius, and shapes beyond documented limits
  are rejected by the host; callers must use a framework fallback.
