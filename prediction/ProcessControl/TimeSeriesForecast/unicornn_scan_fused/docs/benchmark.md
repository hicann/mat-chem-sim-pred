# UnicornnScanFused 性能报告

## 环境

```text
Ascend 910B3
dtype: float32
baseline: torch_npu per-timestep UnICORNN loop
```

## 结果

Shape:

```text
B=32, L=336, enc_in=11, H=64, layers=3
```

| gate | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| component, 1 layer | `79.76 ms` | `0.496 ms` | `160.68x` | probe max diff `5.53e-10` ~ `2.98e-08` |
| E2E, 3-layer encoder | `243.79 ms` | `2.43 ms` | `100.43x` | encoder max diff `2.68e-07` |

## 计时口径

baseline 是 fair torch_npu loop baseline。UnICORNN 的 diagonal nonlinear recurrence 没有原生 NPU primitive，框架路径主要受每步 launch 开销支配；custom kernel 将 `y/z` 状态留在 UB 内按 batch 并行扫描。

原始结果见 `docs/unicornn_scan_fused_e2e_results.json` 和 `docs/unicornn_scan_fused_probe_results.json`。

## TorchAir fullgraph 补充验证

单层主测试：`B=32,L=336,IN=11,H=64`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 4,715 | `250.93 s` | `91.229 ms` | `3.544 ms` | `0.521 ms` | `25.74x` | `6.80x` |

TorchAir/custom 对 eager 的 max diff 分别为 `0/1.90e-07`。`L=32` 时图有 459 个节点，`L=336` 时增至 4,715，说明循环被按时间步展开而非识别为固定大小的 scan kernel。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
