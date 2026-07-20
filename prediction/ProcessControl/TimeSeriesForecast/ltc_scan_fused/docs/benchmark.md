# LtcScanFused 性能报告

## 环境

```text
Ascend 910B3
dtype: float32
baseline: torch_npu per-timestep LTC loop
```

## 结果

Shape:

```text
B=32, L=336, enc_in=11, H=64, layers=3, inner ODE unfolds K=6
```

| gate | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| component, 1 layer | `379.21 ms` | `4.60 ms` | `82.44x` | probe max diff `1.97e-06` ~ `3.92e-06` |
| E2E, 3-layer encoder | `1134.71 ms` | `14.76 ms` | `76.90x` | encoder max diff `2.53e-07` |

## 计时口径

baseline 是 fair torch_npu loop baseline。LTC 每个 timestep 内含 `K=6` 次半隐式 ODE unfold，框架路径会放大逐步 launch 开销；custom kernel 将所有 `K * L` 步融合在一个 kernel 内。

原始结果见 `docs/ltc_scan_fused_e2e_results.json` 和 `docs/ltc_scan_fused_probe_results.json`。

## TorchAir fullgraph 补充验证

单层主测试：`B=32,L=336,IN=11,H=64,K=6`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 24,878 | `616.61 s` | `386.003 ms` | `10.227 ms` | `4.909 ms` | `37.74x` | `2.08x` |

TorchAir/custom 对 eager 的 max diff 都是 `1.34e-07`。每个时间步的 6 次 ODE 更新使完整图膨胀到近 2.5 万节点，首次编译约 10.3 分钟；custom 将 `K*L` 递推保留在一个 kernel 中。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
