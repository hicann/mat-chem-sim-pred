# CfcScanFused 性能报告

## 环境

```text
Ascend 910B3
dtype: float32
baseline: torch_npu per-timestep CfC loop
```

## 结果

Shape：

```text
B=32, L=336, enc_in=11, H=64, layers=3
```

| gate | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| component, 1 layer | `104.39 ms` | `1.23 ms` | `84.83x` | probe max diff `9.87e-08` ~ `2.38e-07` |
| E2E, 3-layer encoder | `330.00 ms` | `4.70 ms` | `70.26x` | encoder max diff `2.33e-07` |

## 计时口径

baseline 是 fair torch_npu loop baseline。CfC recurrence 不能做并行/chunked scan，框架路径会退化成大量小 matmul/activation launch；custom kernel 将 `h` 留在 UB 内按 batch 并行扫描。

原始结果见 `docs/cfc_scan_fused_e2e_results.json` 和 `docs/cfc_scan_fused_probe_results.json`。

## TorchAir fullgraph 补充验证

单层主测试：`B=32,L=336,IN=11,H=64`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 7,062 | `154.54 s` | `118.244 ms` | `8.090 ms` | `1.265 ms` | `14.62x` | `6.40x` |

TorchAir/custom 对 eager 的 max diff 分别为 `1.79e-07/2.38e-07`。TorchAir 显著优化 eager，但仍按时间步展开数千节点；custom 保留运行性能和首次编译成本优势。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
