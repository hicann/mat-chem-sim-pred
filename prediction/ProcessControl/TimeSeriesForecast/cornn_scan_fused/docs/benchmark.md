# CornnScanFused 性能报告

## 环境

```text
Ascend 910B3
dtype: float32
baseline: torch_npu per-timestep coRNN IMEX loop
```

## 结果

Shape：

```text
B=32, L=336, enc_in=11, H=64, layers=3
```

| gate | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| component, 1 layer | `97.96 ms` | `1.73 ms` | `56.73x` | probe max diff `2.33e-09` ~ `3.35e-08` |
| E2E, 3-layer encoder | `338.73 ms` | `6.26 ms` | `54.15x` | encoder max diff `1.49e-07` |

## 计时口径

baseline 是 fair torch_npu loop baseline。coRNN 的二阶非线性 oscillator update 不能改写为线性并行 scan；custom kernel 将 `y/z` 留在 UB 内按 batch 并行推进。

原始结果见 `docs/cornn_scan_fused_e2e_results.json` 和 `docs/cornn_scan_fused_probe_results.json`。

## TorchAir fullgraph 补充验证

单层主测试：`B=32,L=336,IN=11,H=64`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 3,703 | `92.83 s` | `73.529 ms` | `3.579 ms` | `1.718 ms` | `20.54x` | `2.08x` |

TorchAir/custom 对 eager 的 max diff 分别为 `8.94e-08/1.04e-07`。完整图仍按时间步展开，不是单个 scan kernel。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
