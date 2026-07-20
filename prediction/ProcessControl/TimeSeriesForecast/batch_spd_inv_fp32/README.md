# BatchSpdInvFp32 迁移交付目录

本目录从 C:/tslib/deliverables/batch_spd_inv_fp32_experimental 迁移，按 PIDModelFit/CANN_OPERATOR_RULES.md 整理为本仓库时序预测算子交付结构。

## 状态

| 项目 | 状态 |
|------|------|
| msopgen | msopgen/batch_spd_inv_fp32_msopgen.json |
| Ascend C kernel | 已迁入 op_kernel/ |
| op_host | 已迁入 op_host/ |
| tests | 已迁入 tests/ |
| source docs | 已保留到 docs/source_* |

## 说明

完整算法、API、benchmark 和测试状态分别见 docs/algorithm.md、docs/api_reference.md、docs/benchmark.md、docs/test_report.md。

## TorchAir 补充结论

Koopa 原路径的 `torch.linalg.lstsq` 不支持 NPU，运行时回退 CPU；Dynamo/TorchAir 也无法用 `fullgraph=True` 捕获。本算子的价值是补齐设备能力：DMD `B32,m7,E128` 从 `4341.75 ms` CPU fallback 降至 `0.339 ms` 全设备路径，提升 `12823.56x`；Koopa `seq336,pred96,B32` E2E 从 `5300.64 ms` 降至 `8.80 ms`，提升 `602.19x`。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
