# SelectiveScan1D 性能与测试报告

## 测试环境

已完成原型验证环境：

| 项目 | 配置 |
|------|------|
| 机器 | Ascend 910B3 环境 |
| NPU | Ascend 910B3 |
| torch | 2.5.1 |
| torch_npu | 2.5.1 |
| CANN | Ascend 910B3 环境 默认 Ascend toolkit 环境 |
| SOC_VERSION | Ascend910B3 原型构建 |

除本地 Python reference 测试外，迁移目录已在 Ascend 910B3 环境完成 msopgen 构建、ACLNN runtime 和正式 shape benchmark。

## 复现命令

构建：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/TimeSeriesForecast/selective_scan_1d
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B1
cmake --build build -j 2
```

ACL smoke：

```bash
./build/test_aclnn_selective_scan_1d 0
```

NPU benchmark：

```bash
./build/benchmark_selective_scan_1d 0 1 1024 1536 16 10 3
```

Python reference benchmark：

```bash
python tests/benchmark_selective_scan_1d.py \
  --batch 1 --length 128 --dim 64 --state 16 --repeat 3 --warmup 1
```

## Scan-only benchmark

Shape:

```text
B=1,D=1536,N=16,L=1024
```

| Path | Latency | 说明 |
|------|--------:|------|
| CPU naive scan | `1025.7 ms` | Python/CPU reference |
| NPU torch_npu naive scan | `237.7 ms` | 框架 fallback，Python loop + 多小算子 |
| NPU custom scalar | `41.88 ms` | 第一版自定义算子 |
| NPU custom vectorized | `27.38 ms` | UB + vector ops + hardware `Exp` |

## Mamba block E2E

Shape:

```text
B=1,L=1024,D_MODEL=768,D_INNER=1536,N=16
```

只替换一个算子类型：`SelectiveScan1D`。其他 block 组件仍使用 torch_npu 框架算子。

| Version | E2E mean | scan time | 输出校验 |
|---------|---------:|----------:|----------|
| torch_npu naive scan block | `185.72 ms` | `185.667 ms` | `out_sum=-20.945343` |
| custom SelectiveScan1D block | `20.59 ms` | `19.833 ms` | `out_sum=-20.945343` |

```text
E2E speedup = 9.02x
scan speedup = 9.36x
max_abs_diff = 2.794e-09
mean_abs_diff = 1.943e-10
```

## MTO forecasting E2E

Scope:

```text
real MTO CSV windows -> minimal Mamba forecasting model -> forecast head -> MSE/MAE
```

Shape:

```text
B=1, seq_len=1024, pred_len=24, input_features=64,
D_MODEL=768, D_INNER=1536, N=16, validation windows=8
```

| Path | Validation loop | Per window | Throughput | MSE | MAE |
|------|----------------:|-----------:|-----------:|----:|----:|
| torch_npu naive scan | `1699.62 ms` | `212.45 ms` | `4.71 windows/s` | `7.10298e+09` | `7320.89` |
| custom SelectiveScan1D | `166.38 ms` | `20.80 ms` | `48.08 windows/s` | `7.10298e+09` | `7320.89` |

```text
Validation-loop speedup = 10.22x
Scan-component speedup = 10.45x
MSE/MAE absolute diff = 0.000e+00 / 0.000e+00
Prediction max/mean abs diff = 1.431e-06 / 1.314e-07
```

## 计时口径

| 口径 | 说明 |
|------|------|
| `CPU reference` | CPU/NumPy 或 C++ reference 递推耗时 |
| `NPU hot` | 输入已在 device，使用 ACL event 统计重复 launch 平均耗时 |
| `NPU cold` | 可在后续 benchmark 中扩展为 H2D + compute + D2H |
| `E2E loop` | 模型验证循环中的稳定推理耗时，不含训练 |

## 风险与后续验证

- 当前迁移代码已完成 CANN 构建、runtime 和 benchmark 验证。
- `state` 变大时 UB buffer 规模会变大，需要补充多 shape 覆盖。
- 后续可追加 `B>1`、更长 `L`、更多 feature/channel 的 benchmark 表。

## TorchAir fullgraph 补充验证

scan-only 主测试：`B=1,L=1024,D=1536,N=16`，`backend="npu"`、`fullgraph=True`、`dynamic=False`。

| FX 节点 | 首次编译+运行 | eager | TorchAir | custom | 图/eager | custom/图 |
|---:|---:|---:|---:|---:|---:|---:|
| 18,441 | `883.51 s` | `343.825 ms` | `27.952 ms` | `20.077 ms` | `12.30x` | `1.39x` |

TorchAir/custom 对 eager 的 max diff 分别为 `0/4.66e-09`。TorchAir 能完整成图，但不是固定大小的 scan 节点，首次编译约 14.7 分钟。该结果与上面的历史 benchmark 是独立补充运行，不应混用绝对时间计算新倍率。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
