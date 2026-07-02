# PidControlPerformanceMetrics 测试报告

## 测试环境

- NPU 环境：node202
- CANN：Ascend Toolkit `/usr/local/Ascend/ascend-toolkit/latest`
- SOC：Ascend910B3
- NPU device：3
- CPU 对比线程数：64
- 数据类型：float32

## 测试口径

- `npu_kernel`：输入已在 Device 上，仅统计算子执行时间。
- `npu_e2e`：包含 H2D 输入拷贝、kernel 执行、D2H 输出拷贝。
- `cpu_parallel`：CPU 多线程持久线程池，一次扫描融合计算 20 个指标。

该算子主要用于 NPU 侧整定/仿真/评分流水线的融合后处理。若数据来自 CPU 且只单独调用一次，端到端收益需要单独评估。

## 准确度

NPU 输出与 CPU double 参考比较，最大相对误差保持在 `1e-4` 以内。最大绝对误差通常出现在 `ITAE`，因为该指标会随时间权重和窗口长度累加到 `1e7` 量级，float32 累加的绝对误差会被放大，但相对误差仍较小。

| B | N | max_abs_err | abs err metric | max_rel_err |
|---:|---:|---:|---|---:|
| 128 | 4096 | 6.5 | ITAE | 4.83386e-5 |
| 512 | 4096 | 6.75 | ITAE | 5.11775e-5 |
| 1024 | 8192 | 46 | ITAE | 7.29049e-5 |
| 2048 | 4096 | 7.25 | ITAE | 5.26023e-5 |
| 2048 | 8192 | 47 | ITAE | 7.49399e-5 |

## 性能结果

| B | N | CPU single ms | CPU 64T ms | NPU kernel ms | NPU e2e ms | kernel / CPU 64T | e2e / CPU 64T |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4096 | 5.91447 | 0.464842 | 0.0281688 | 0.759054 | 16.502x | 0.612x |
| 512 | 4096 | 24.1734 | 0.993433 | 0.0462995 | 3.00491 | 21.457x | 0.331x |
| 1024 | 8192 | 96.5096 | 3.87055 | 0.0552966 | 14.4265 | 69.996x | 0.268x |
| 2048 | 4096 | 131.101 | 4.77999 | 0.0554786 | 9.64383 | 86.159x | 0.496x |
| 4096 | 4096 | 258.319 | 7.50180 | 0.0935643 | 7.74675 | 80.178x | 0.968x |
| 2048 | 8192 | 251.823 | 10.1350 | 0.100558 | 11.0751 | 100.788x | 0.915x |

## 结论

`PidControlPerformanceMetrics` 相比单独的 Cpk/Ppk 指标更适合 NPU：一次扫描融合 20 个指标，kernel-only 相对 64 线程 CPU 有约 `16x` 到 `100x` 加速。

端到端结果仍受 H2D/D2H 拷贝主导。对于 `4096 x 4096` 和 `2048 x 8192` 这类较大规模，NPU e2e 已接近 CPU 64 线程；如果 `pv/sp` 来自上游 NPU 仿真或模型评分并保持 device-resident，收益会更明显。

因此该算子的推荐定位不是“CPU 侧单独调用一次指标计算”，而是“整定/仿真/评分 NPU 流水线的融合后处理算子”。

## 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_control_performance_metrics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH"

./build/test_aclnn_pid_control_performance_metrics 3
./build/benchmark_pid_control_performance_metrics 3 128 4096 20 3 64
./build/benchmark_pid_control_performance_metrics 3 512 4096 10 2 64
./build/benchmark_pid_control_performance_metrics 3 1024 8192 5 2 64
./build/benchmark_pid_control_performance_metrics 3 2048 4096 10 2 64
./build/benchmark_pid_control_performance_metrics 3 4096 4096 3 1 64
./build/benchmark_pid_control_performance_metrics 3 2048 8192 3 1 64
```
