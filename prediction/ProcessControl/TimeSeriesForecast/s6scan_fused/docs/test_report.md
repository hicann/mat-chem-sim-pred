# S6scanFused 测试报告

## 来源验证

| 项目 | 状态 |
|------|------|
| Ascend C kernel | 已迁入 |
| msopgen JSON | 本次按通用 scan schema 补齐 |
| op_host / tiling | 本次按统一 scan schema 补齐 |
| C++ probe | 源交付物未提供，保留 Python ctypes probe |
| Python E2E | 已迁入 |
| component speedup | `136.56x` |
| E2E speedup | `44.15x` |

## 本次迁移

已迁入或补齐：

- op_kernel/s6scan_fused_kernel.cpp
- op_host/s6scan_fused_host.cpp
- op_host/s6scan_fused_tiling.h
- msopgen/s6scan_fused_msopgen.json
- docs/algorithm.md
- docs/api_reference.md
- docs/benchmark.md
- tests/

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark。

## TorchAir fullgraph 验证

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B32,L336,IN11,H64` |
| 原模型完整成图 | FAIL，缺 `aten.softplus.default` converter |
| 人工等价改写 | PASS，7,069 FX 节点 |
| 首次编译+运行 | `141.83 s` |
| TorchAir / custom | `4.372 / 0.674 ms` |
| custom vs TorchAir | `6.48x` |
| custom max diff | `4.47e-08` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
