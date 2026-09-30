<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- Clean Ascend910B3 Release build: passed.
- Deterministic, random, stable-tie, dispatch, and Host-contract tests:
  5 passed.
- ACL smoke output matches the 12-value CPU reference exactly.
- MUTAG checkpoint SHA256:
  `c0eaeb3b41512354fcb4a347dbdd225e052bf75e43ba52b3ae43300145f40f9f`.
- Held-out checkpoint accuracy: `0.8947`.
- Four model shapes: maximum pool error `1.20e-7`, maximum model error
  `2.87e-6`, prediction and task metrics unchanged.

The full-model benchmark includes all graph convolutions, sort aggregation,
and classifier layers. It does not substitute graph counts for model E2E or
include host/device transfer time in only one side.
