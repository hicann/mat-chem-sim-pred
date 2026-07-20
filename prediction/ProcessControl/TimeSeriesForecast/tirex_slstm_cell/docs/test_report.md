# TirexSlstmCell 测试报告

## 本次迁移

| 项目 | 状态 |
|------|------|
| kernel | 已迁入 |
| op_host | 已迁入 |
| msopgen | 已迁入 |
| tests/e2e | 已迁入源侧脚本 |
| CANN runtime 复验 | Ascend 910B3 PASS |

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark。源侧 E2E 脚本继续保留在 `tests/`。

## 正确性与性能证据

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B64,S64,H512,heads4` |
| custom vs reference max diff | `1.45e-06` |
| torch_npu / custom cell | `52.349 / 3.658 ms`，`14.31x` |
| torch_npu / custom layer | `53.910 / 5.153 ms`，`10.46x` |
| 原模型 TorchAir fullgraph | FAIL，缺 `aten.log_sigmoid_forward.default` converter |
| 人工等价改写 | PASS，2,888 FX 节点 |
| TorchAir / custom | `6.318 / 3.542 ms`，custom 快 `1.78x` |
| 正式复验 max diff | `7.45e-08` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
