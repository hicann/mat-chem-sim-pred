# S6scanFused 时序 scan 融合算子

本目录从 C:\tslib\deliverables\s6scan_fused_experimental 迁移，按 PIDModelFit/CANN_OPERATOR_RULES.md 整理为 CANN 算子交付结构。

## 算子入口

```text
S6scanFused(x:[B,L,IN], weight:[*,H], bias:[*]) -> output:[B,L,H]
```

具体递推公式和权重打包方式见 docs/algorithm.md，原始交付说明保存在 docs/source_readme.md。

## 当前证据

| 项目 | 结果 |
|------|------|
| component speedup | `136.56x` |
| E2E speedup | `44.15x` |
| 迁移成熟度 | Python ctypes probe + Python E2E |

## 文件结构

- op_kernel/s6scan_fused_kernel.cpp - Ascend C fused scan kernel
- op_host/s6scan_fused_host.cpp - op-host tiling / shape inference / OpDef
- op_host/s6scan_fused_tiling.h - tiling data definition
- msopgen/s6scan_fused_msopgen.json - msopgen IR
- tests/ - source probe / E2E scripts
- docs/ - algorithm、API、benchmark、test report

## TorchAir 补充结论

原模型在主测试 `B32,L336,IN11,H64` 上缺少 `aten.softplus.default` converter，无法直接完整成图。将 softplus 人工等价改写为稳定的基础张量操作后，TorchAir 为 `4.372 ms`，custom 为 `0.674 ms`，custom 仍快 `6.48x`；首次编译和运行耗时 `141.83 s`，max diff `4.47e-08`。人工改写结果不代表 TorchAir 能自动优化原模型。custom 直接实现 S6 门控递推语义，接入时不需要为了 converter 缺失重写 softplus 表达。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
