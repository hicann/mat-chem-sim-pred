# BatchSpdInvFp32 测试报告

## 本次迁移

| 项目 | 状态 |
|------|------|
| kernel | 已迁入 |
| op_host | 已迁入 |
| msopgen | 已迁入 |
| tests/e2e | 已迁入源侧脚本 |
| CANN runtime 复验 | Ascend 910B3 PASS |

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark。源侧可运行脚本保留在 `tests/` 和 `scripts/`。

## 正确性与性能证据

| 项目 | 结果 |
|---|---|
| DMD K vs `torch.linalg.lstsq` | max diff `< 7e-08` |
| DMD B32 m7 E128 | CPU fallback/custom `4341.75/0.339 ms`，`12823.56x` |
| Koopa seq336 E2E | CPU fallback/custom `5300.64/8.80 ms`，`602.19x` |
| TorchAir fullgraph | FAIL，NPU 不支持 `linalg_lstsq`，且动态输出 shape 无法捕获 |
| 口径说明 | 原 NPU 路径回退 CPU，已明确标注 |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
