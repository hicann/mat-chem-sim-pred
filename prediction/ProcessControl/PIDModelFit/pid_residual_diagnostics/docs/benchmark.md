# PidResidualDiagnostics Benchmark Report

## 测试环境

- NPU 环境：node202
- SOC：Ascend910B3
- NPU device：3
- CANN：`/usr/local/Ascend/ascend-toolkit`
- CPU 对比线程数：64
- 数据类型：float32

## 测试口径

- `cpu_parallel`：CPU 64 线程，计算 8 个残差指标和 `max_lag` 个自相关。
- `npu_kernel`：输入已在 Device，仅统计算子执行时间。
- `npu_e2e`：包含 `actual/predicted` H2D、kernel、`metrics/autocorr` D2H。

当前输出指标：

```text
mean_residual,std_residual,mae,rmse,max_abs_residual,fit_percent,durbin_watson,ljung_box_q
```

当前版本固定输出 8 个指标，使用 `fit_percent` 作为拟合优度指标。数值和性能结论均基于该 8 指标版本。

## 正确性

Smoke：

```text
PidResidualDiagnostics smoke mean=0 mae=0.5 rmse=0.707107 max_abs=1 dw=1.5 autocorr=[0, -0.5]
PASSED
```

Benchmark 中 CPU/NPU 输出对比：

- `max_autocorr_abs_err = 0`
- `max_metric_abs_err <= 0.0054`
- 最大 metric 误差来自 `fit_percent`，相对误差约 `5.8e-5`

## 性能结果

| B | N | max_lag | CPU 64T ms | NPU kernel ms | NPU e2e ms | kernel/CPU64T | e2e/CPU64T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1024 | 16 | 0.468949 | 0.092059 | 0.442651 | 5.09x | 1.06x |
| 256 | 1024 | 32 | 1.39999 | 0.0552526 | 0.359782 | 25.34x | 3.89x |
| 512 | 2048 | 32 | 2.33311 | 0.075771 | 0.957573 | 30.79x | 2.44x |
| 1024 | 2048 | 64 | 8.62803 | 0.0683055 | 1.85133 | 126.31x | 4.66x |

## 结论

`PidResidualDiagnostics` 是本次正式提交的模型诊断算子之一：

- 业务价值明确：用于 PID 模型辨识后的残差白噪声检查、拟合质量评估和模型可信度判断。
- 计算结构适合 NPU：批量归约、多 lag 相关性扫描、输出小。
- e2e 也能超过 CPU 64 线程，说明即使输入来自 Host，该算子在中大规模下仍有实际加速空间。

## 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_residual_diagnostics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/8.1/lib64:${LD_LIBRARY_PATH:-}"

./build/test_aclnn_pid_residual_diagnostics 3
./build/benchmark_pid_residual_diagnostics 3 128 1024 16 5 1 64
./build/benchmark_pid_residual_diagnostics 3 256 1024 32 5 1 64
./build/benchmark_pid_residual_diagnostics 3 512 2048 32 3 1 64
./build/benchmark_pid_residual_diagnostics 3 1024 2048 64 2 1 64
```
