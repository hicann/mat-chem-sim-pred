# PidWindowedResidualDiagnostics Delivery Note

## 结论

`PidWindowedResidualDiagnostics` 是 PID 模型辨识后的窗口化残差诊断算子。它不是新的模型拟合算子，而是模型可信度检查和上线质量门禁算子：在 `actual` 与 `predicted` 已经得到后，按滑动窗口检查每段残差是否白噪声、是否存在局部漂移、周期振荡或分段模型失配。

## 解决什么问题

全局残差诊断只能给出整段数据的平均拟合质量，可能出现这种情况：

```text
整段 RMSE/fit percent 看起来合格，但某几个时间段已经明显漂移或振荡。
```

窗口化诊断把每条回路拆成多个时间窗口，对每个窗口独立输出：

```text
metrics[B, W, 8]
autocorr[B, W, max_lag]
```

这样工程侧可以定位到具体窗口，而不是只看到一个全局平均结论。

## 典型使用场景

- PID 模型辨识后，检查 FOPDT/IPDT/SOPDT 模型是否在整段数据上稳定有效。
- 在线预测或数字孪生流水线中，对 device-resident 的 `actual/predicted` 结果做 NPU 侧质量门禁。
- 对批量回路做巡检，找出局部漂移、周期性残差、自相关过强或局部拟合失配的回路和时间段。
- 为后续重辨识、模型降级、报警或人工复核提供窗口级证据。

## 指标输出

每个窗口输出 8 个指标，顺序与 `PidResidualDiagnostics` 保持一致：

```text
0 mean_residual
1 std_residual
2 mae
3 rmse
4 max_abs_residual
5 fit_percent
6 durbin_watson
7 ljung_box_q
```

同时输出 `autocorr[B, W, max_lag]`，用于查看残差在多个 lag 上的相关性。

## 本次交付

- `op_host/pid_windowed_residual_diagnostics_host.h`
- `op_host/pid_windowed_residual_diagnostics_host.cpp`
- `op_host/pid_windowed_residual_diagnostics_def.cpp`
- `op_kernel/pid_windowed_residual_diagnostics_kernel.cpp`
- `examples/test_aclnn_pid_windowed_residual_diagnostics.cpp`
- `tests/benchmark_pid_windowed_residual_diagnostics_aclnn.cpp`
- `CMakeLists.txt`
- `README.md`
- `docs/algorithm.md`
- `docs/api_reference.md`
- `docs/benchmark.md`

## 验收结果

本地 Python reference：

```text
7 passed
```

node202 / Ascend910B3：

```text
CANN build: PASSED
ACLNN smoke: PASSED
```

Smoke 输出：

```text
PidWindowedResidualDiagnostics smoke windows=2 w0_mean=0 w0_mae=0.5 w0_rmse=0.707107 w0_dw=1.5 w0_autocorr=[0, -0.5]
PASSED
```

Benchmark 结果：

```text
B=64 N=2048 windows=15 window=256 stride=128 lag=16
cpu_64T_ms=4.69239
npu_kernel_ms=0.0470184
npu_resident_e2e_ms=0.0727348
metric_max_abs=7.62939e-06
autocorr_max_abs=0

B=128 N=4096 windows=15 window=512 stride=256 lag=32
cpu_64T_ms=6.07111
npu_kernel_ms=0.0818743
npu_resident_e2e_ms=0.195025
metric_max_abs=7.62939e-06
autocorr_max_abs=0
```

## 边界说明

- 当前 `resident_e2e` 假设 `actual/predicted` 已在 Device，仅统计 kernel 和输出 D2H。
- 如果输入来自 Host，还需要补 cold H2D 端到端数据。
- 当前 kernel 是 GlobalTensor 标量访问原型，后续可以继续做 UB staging 和 lag 维度访存优化。
- 极小窗口不是当前主推荐场景，容易被 launch 和调度开销主导。
