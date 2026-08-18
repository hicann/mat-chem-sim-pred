<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# Test Report

- NumPy reference and 12 contract tests cover the four cross-sign outputs,
  zero-degree rows, CSR bounds, channel alignment, dtypes, shapes, and
  non-finite inputs.
- ACL smoke rejects an unsupported channel count and validates real-device
  output against the CPU formula.
- Release builds cover Ascend C kernel, host API, OpDef, and ACL smoke.
- Complete SignedGCN benchmarks use the strongest correct native index-add
  baseline at three scales and 100 alternating samples per path.
- Level1 traces attribute only the second-layer cross-sign pack.
- Numerical and task parity are reported against both official self drift and a
  controlled deterministic replacement-equivalent path.

Raw JSON evidence is under `tests/model_e2e`.

## Final Evidence Binding

```text
model_source_revision=pytorch_geometric@003c3cd8a10520567ceaeda619f0315e30ec2f66
custom_source_tree_status=local uncommitted handoff; no commit or push
custom_code_manifest_sha256=eebb58c794dfb9dc38607a01d8c8a8de5728b33342f7ba4cfe352e9c2fa346bd
clean_build_dir=/home/huawei/hot_model_15b_20260802/formal_final/csr_signed_cross_mean_pack_fused/build_verify_license_20260803
signedgcn_bitcoin_e2e.json=07ab1cd8071ec9b7006ae8975067326425ff77e8570be5c204a35b2d3b1a35b2
signedgcn_bitcoin_e2e_repeat2.json=25af07a0553475c3fe1af08f2b6e69258239f6016ef6e5e6b0097ada2824e8cd
signedgcn_bitcoin_hotspot.json=5e66ceb1e0e1a169917e5082461692eb692c51b467b4306311c71b62560db2bc
```

The manifest digest is SHA-256 over sorted `sha256  relative/path` lines for
`CMakeLists.txt`, `op_host`, `op_kernel`, reference code, tests, and model
scripts, excluding Markdown, JSON, and generated caches.
