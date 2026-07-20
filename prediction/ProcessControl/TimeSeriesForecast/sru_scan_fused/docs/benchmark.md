# SruScanFused 性能报告

## 环境

```text
Ascend 910B3
dtype: float32
baseline: torch_npu per-timestep SRU loop
```

## 结果

Shape:

```text
B=32, L=336, enc_in=11, H=64, layers=3
```

| gate | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| component, 1 layer | `128.06 ms` | `0.78 ms` | `163.48x` | probe max diff `2.24e-08` ~ `3.73e-08` |
| E2E, 3-layer encoder | `387.41 ms` | `5.42 ms` | `71.46x` | encoder max diff `1.04e-07` |

## 计时口径

baseline 是 fair torch_npu loop baseline。SRU 的 gate 是 input-only projection 加 peephole，逐步 recurrence 极度 launch-bound；custom kernel 将 `c` 状态留在 UB 内按 batch 并行扫描。

原始结果见 `docs/sru_e2e_results.json`。

## TorchAir fullgraph 补充验证

单层主测试：`B=32,L=336,IN=11,H=64`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 7,069 | `165.43 s` | `118.829 ms` | `5.272 ms` | `0.884 ms` | `22.54x` | `5.97x` |

TorchAir/custom 对 eager 的 max diff 分别为 `0/1.79e-07`。完整图显著减少 eager 调度开销，但 custom 仍保持近 6 倍运行优势。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
