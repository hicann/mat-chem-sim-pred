<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
Licensed under the CANN Open Software License Agreement Version 2.0.
-->

# CsrSignedCrossMeanPackFused

Fuses the maintained PyTorch Geometric `SignedConv(first_aggr=False)` cross-sign
mean pack used by the second layer of `SignedGCN`. One CSR traversal produces
the four positive/negative neighborhood means consumed by the layer.

The package contains an Ascend C kernel, ACLNN-style host API, OpDef, NumPy
reference, contract tests, real-device ACL smoke, explicit native fallback,
complete-model profiler evidence, and a trained Bitcoin-OTC regression.

## Value

- Exact source: PyG `torch_geometric/nn/conv/signed_conv.py` at commit
  `003c3cd8a10520567ceaeda619f0315e30ec2f66`.
- Conservative two-run component speedup: `3.757x / 5.332x / 6.273x`.
- Conservative two-run E2E reduction: `47.49% / 51.91% / 53.21%`.
- Exact replaceable-stage hotspot: `50.83% / 57.70% / 60.75%`.
- Real task: Bitcoin-OTC, 6,005 nodes; checkpoint AUC `0.8073`, F1 `0.9072`.
- Controlled embedding max error `2.98e-8`; prediction agreement rounds to 100%.

See `docs/benchmark.md` and `tests/model_e2e/README.md` for raw evidence and
reproduction commands.
