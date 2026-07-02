# PidWindowedResidualDiagnostics

## 这个算子是干啥的

`PidWindowedResidualDiagnostics` 是 PID 模型辨识后的窗口化残差诊断算子。它把每条回路的 `actual/predicted` 时间序列切成多个滑动窗口，并在每个窗口内计算残差统计量和多 lag 自相关，用来回答一个工程问题：

```text
模型整体看起来拟合不错，但在某些时间段是否已经漂移、振荡或局部失配？
```

输入和输出：

```text
actual[B, N], predicted[B, N]
    -> metrics[B, W, 8]
    -> autocorr[B, W, max_lag]
```

其中 `W = 1 + (N - window_size) / stride`。每个窗口输出 8 个指标：残差均值、标准差、MAE、RMSE、最大残差、fit percent、Durbin-Watson 和 Ljung-Box 风格统计量。

## window 和 stride 是什么

`window_size` 是每个局部诊断窗口包含的采样点数，`stride` 是相邻窗口起点之间的步长。`stride` 不是交叠样本数本身；相邻窗口交叠为：

```text
overlap = window_size - stride
```

例如：

```text
N = 4096
window_size = 512
stride = 256
```

窗口为：

```text
window 0: [0, 512)
window 1: [256, 768)
window 2: [512, 1024)
...
```

这里相邻窗口 50% 重叠。`stride` 越小，时间定位越细、计算越多；`stride` 越大，计算越少，但可能漏掉短时异常。

## 它不是全局指标平均

窗口化残差诊断不是把全局指标再平均一遍，而是给每个窗口独立输出一套指标：

```text
metrics[B, W, 8]
autocorr[B, W, max_lag]
```

它用于定位局部问题，例如：

```text
哪个窗口 RMSE 最大？
哪个窗口 max_abs_residual 最大？
哪个窗口 mean_residual 持续偏正/偏负？
哪个窗口 Durbin-Watson 偏离 2？
哪个窗口 Ljung-Box Q 或 autocorr 很大，说明存在局部振荡/周期性残差？
```

全局残差诊断回答“整段模型是否总体可信”；窗口化残差诊断回答“是否存在某一段局部漂移、局部震荡或分段模型失配”。

## 交付内容

本目录已从 Python reference 尝试池推进为可构建、可运行的 Ascend C 原型算子，包含：

- `aclnnPidWindowedResidualDiagnostics` host API。
- `pid_windowed_residual_diagnostics_kernel` device kernel。
- ACLNN smoke 示例。
- ACLNN benchmark。
- Python reference、Python 单测和本机 benchmark。
- 算法、API、benchmark 和交付说明文档。

## 为什么值得做

- 全局残差诊断只能说明整段数据的平均拟合质量，容易掩盖短时间漂移、局部振荡和分段模型失配。
- 窗口化后每个 `(batch, window)` 独立归约，计算结构是批量扫描和多 lag 自相关，适合 NPU 并行。
- 输出只有 `metrics[B,W,8]` 和 `autocorr[B,W,L]`，相比回传完整窗口轨迹，D2H 压力小。
- 可作为 FOPDT/IPDT/SOPDT 模型辨识或 NPU 侧预测流水线后的 device-side 质量门禁。

## 已验证结果

- 本地 Python reference：`7 passed`。
- node202 / Ascend910B3 CANN 构建通过。
- ACLNN smoke 通过。
- node202 benchmark 数值与 CPU reference 对齐，`metric_max_abs <= 7.63e-6`，`autocorr_max_abs = 0`。
- 典型规模 `B=128,N=4096,W=15,window=512,lag=32` 下，CPU 64T `6.07111 ms`，NPU kernel `0.0818743 ms`，resident e2e `0.195025 ms`。

## 本机验证

```bash
python -m pytest prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics/tests/test_pid_windowed_residual_diagnostics.py -q
python prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics/tests/benchmark_pid_windowed_residual_diagnostics.py
```

## 文档

- [算法说明](docs/algorithm.md)
- [API 说明](docs/api_reference.md)
- [Benchmark Note](docs/benchmark.md)
- [交付说明](docs/delivery.md)

## Ascend C 构建与 smoke

```bash
cd prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
./build/test_aclnn_pid_windowed_residual_diagnostics 0
./build/benchmark_pid_windowed_residual_diagnostics 0 128 4096 512 256 32 5 64
```
