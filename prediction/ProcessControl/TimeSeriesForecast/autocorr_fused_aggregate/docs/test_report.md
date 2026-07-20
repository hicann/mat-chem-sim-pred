# AutoCorrFusedAggregate 测试报告

## 已完成

来自 `C:\tslib\deliverables\route_b_custom_ops\AutoCorrFusedAggregate` 的 Ascend 910B3 验证：

| 项目 | 结果 |
|------|------|
| msopgen JSON | 已提供 |
| Ascend C kernel | 已提供 |
| op_host tiling / InferShape / OpDef | 已提供 |
| ACLNN smoke | PASS |
| component benchmark | PASS |
| Autoformer-style forecasting E2E | PASS |
| validation-loop speedup | `10.06x` |
| aggregation speedup | `28.06x` |
| prediction max abs diff | `5.960e-08` |

## 本次迁移

已迁入：

- `op_kernel/autocorr_fused_aggregate_kernel.cpp`
- `op_host/autocorr_fused_aggregate_host.cpp`
- `op_host/auto_corr_fused_aggregate_tiling.h`
- `msopgen/autocorr_fused_aggregate_msopgen.json`
- `examples/test_aclnn_autocorr_fused_aggregate.cpp`
- `tests/autocorr_fused_aggregate_smoke.cpp`
- `tests/benchmark_autocorr_fused_aggregate_aclnn.cpp`
- `tests/autoformer_autocorr_e2e_ctypes.py`

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime 和正式 shape benchmark：

```bash
cd prediction/ProcessControl/TimeSeriesForecast/autocorr_fused_aggregate
msopgen gen -i msopgen/autocorr_fused_aggregate_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_autocorr_fused_aggregate -lan cpp
```

## TorchAir fullgraph 验证

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B1,heads4,embed16,L192,top_k3` |
| 原模型 fullgraph | FAIL，缺 `aten.roll.default` converter |
| 人工等价改写 | PASS，1,546 FX 节点 |
| TorchAir / custom | `4.568 / 0.618 ms` |
| custom vs TorchAir | `7.39x` |
| 模型量级 max diff | `7.45e-09` |
| 宽动态范围 max diff | `1.43e-05` |

原模型不能直接 TorchAir 成图；上表 TorchAir 时间来自人工 `slice + cat` 等价改写。custom 直接实现原目标语义，不需要维护该框架改写。

最终 FastExp 正负指数缩放分支已重新完成 msopgen/C++ 构建和 ACLNN runtime：

| 复验输入 | custom median | max diff |
|---|---:|---:|
| 模型量级，scale `0.03` | `0.610 ms` | `7.45e-09` |
| 宽动态范围，scale `1.0` | `0.633 ms` | `1.43e-05` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
