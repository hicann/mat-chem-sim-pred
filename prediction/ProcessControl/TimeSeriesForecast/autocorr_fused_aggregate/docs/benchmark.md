# AutoCorrFusedAggregate 性能报告

## 环境

```text
Ascend 910B3 / CANN 8.1.RC1 / torch_npu 2.5.1
dtype: float32
baseline: torch_npu Autoformer-style autocorrelation aggregation
```

## 结果

Forecasting E2E shape:

```text
windows=8, seq_len=192, pred_len=24, D_MODEL=64, heads=4, embed=16, top_k=3
```

| 项目 | 框架路径 | 自定义算子路径 | speedup | 精度 |
|------|----------|----------------|---------|------|
| validation loop | `144.36 ms` | `14.35 ms` | `10.06x` | MSE/MAE diff `0` |
| per-window | `18.05 ms` | `1.79 ms` | `10.06x` | prediction max diff `5.96e-08` |
| autocorr aggregation | `18.4998 ms` | `0.6593 ms` | `28.06x` | same output path |
| kernel hot microbench | CPU reference vs NPU hot | `1.793 ms` NPU hot | `43.99x` vs CPU | smoke PASS |

## Scaling

Window count scaling shows steady-state model loop speedup around `10x-16x`; all-in speedup including CSV preprocessing rises from `1.10x` at 1 window to `1.73x` at 16 windows.

原始结果见 `docs/autocorr_autoformer_e2e_report.md` 和 `docs/autocorr_autoformer_scaling_report.md`。

## TorchAir fullgraph 补充验证

正式测试使用 `B=1,heads=4,embed=16,L=192,top_k=3`：

| 路径 | 原模型 fullgraph | FX 节点 | 首次编译+运行 | eager | TorchAir | custom | custom/图 |
|---|---|---:|---:|---:|---:|---:|---:|
| 原 `roll` 表达 | FAIL，缺 converter | 784 | 不适用 | 17.900 ms | 不适用 | - | - |
| 人工 `slice + cat` 等价改写 | PASS | 1,546 | 69.70 s | 24.324 ms | 4.568 ms | 0.618 ms | 7.39x |

人工改写与原 `roll` 公式逐元素一致，但需要修改模型代码，不能视为 TorchAir 自动优化；这反映当前 `aten.roll` converter 覆盖不足。custom 直接实现目标语义，只需接入 ACLNN 调用，不需要维护 `slice + cat` 改写。custom 计时包含 ACLNN 描述符、执行器、kernel 和同步开销。

模型量级输入的 custom max diff 为 `7.45e-09`；标准差为 1 的宽动态范围输入 max diff 为 `1.43e-05`。最终 FastExp 正负指数缩放分支在 910B3 重建后复验为 `0.610 ms / 7.45e-09` 和 `0.633 ms / 1.43e-05`。表中 `0.618 ms` 是与 TorchAir 同轮配对测试值。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
