# PidProcessCapabilityMetrics 测试报告

## 测试环境

- NPU 环境：node202
- CANN：Ascend Toolkit `/usr/local/Ascend/ascend-toolkit/latest`
- SOC：Ascend910B3
- NPU device：3
- CPU 对比线程数：64
- 数据类型：float32

## 测试数据

benchmark 使用确定性合成过程数据：

- `values[B, N]`：以 50 为中心，叠加不同回路 drift、scale 和周期扰动。
- `lsl[B]`：固定为 47。
- `usl[B]`：固定为 53。
- 少量样本注入越限点，用于验证 `out_of_spec_count` 和 `out_of_spec_ratio`。

输出指标顺序：

```text
mean,std_sample,std_population,cp,cpu,cpl,cpk,pp,ppk,out_ratio,out_count,min,max
```

## 对比口径

CPU 侧提供两类基线：

- `cpu_parallel`：Welford 稳定方差，多线程持久线程池。
- `cpu_fast_parallel`：`sum/sum_sq` 快速方差，多线程持久线程池。

NPU 侧提供两类时间：

- `npu_kernel`：输入已在 Device 上，仅统计算子执行时间。
- `npu_e2e`：包含 H2D 输入拷贝、kernel 执行、D2H 输出拷贝。

`npu_kernel` 更适合评价算子本身；`npu_e2e` 更适合评价 CPU 数据源直接调用的整体收益。

## 准确度

NPU 输出与 CPU Welford 参考最大绝对误差：

| B | N | max_abs_err |
|---:|---:|---:|
| 128 | 4096 | 1.41144e-4 |
| 512 | 4096 | 1.41144e-4 |
| 1024 | 8192 | 2.86102e-4 |
| 2048 | 4096 | 1.44958e-4 |
| 4096 | 4096 | 1.56403e-4 |
| 2048 | 8192 | 2.86102e-4 |

误差来自 float32 累加和 NPU kernel 内部近似开方，满足过程能力指标巡检场景的精度要求。

## 性能结果

| B | N | CPU Welford 64T ms | CPU fast 64T ms | NPU kernel ms | NPU e2e ms | kernel / CPU fast | e2e / CPU fast |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 4096 | 0.475654 | 0.365382 | 0.172293 | 0.794343 | 2.12070x | 0.45998x |
| 512 | 4096 | 1.990270 | 0.645767 | 0.177989 | 1.083030 | 3.62813x | 0.59626x |
| 1024 | 8192 | 2.788000 | 1.209490 | 0.290623 | 4.390670 | 4.16172x | 0.27547x |
| 2048 | 4096 | 2.732180 | 1.127160 | 0.191054 | 1.978900 | 5.89967x | 0.56959x |
| 4096 | 4096 | 5.781140 | 2.180860 | 0.190339 | 20.712800 | 11.45780x | 0.10529x |
| 2048 | 8192 | 4.963120 | 1.888250 | 0.304033 | 12.050600 | 6.21066x | 0.15669x |

## 结论

`PidProcessCapabilityMetrics` 的 kernel-only 性能相对 CPU 多线程有明确优势：相对更苛刻的 `cpu_fast_parallel` 基线仍有约 2.1x 到 11.5x 加速。

如果输入数据已经位于 NPU，或者 Cpk/Ppk 是 NPU 侧整定、仿真、评分流水线中的一个融合后处理步骤，该算子有工程价值。

如果输入来自 CPU 且只做一次 Cpk/Ppk 计算，端到端收益取决于 H2D/D2H 拷贝成本，数据搬移可能抵消 kernel 优势。此时建议将该算子作为 NPU 流水线的一环使用，或继续做多窗口批处理与上游算子融合，减少主机与设备之间的数据搬移。

## 复现命令

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_process_capability_metrics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:$LD_LIBRARY_PATH"

./build/test_aclnn_pid_process_capability_metrics 3
./build/benchmark_pid_process_capability_metrics 3 128 4096 20 3 64
./build/benchmark_pid_process_capability_metrics 3 512 4096 10 2 64
./build/benchmark_pid_process_capability_metrics 3 1024 8192 5 2 64
./build/benchmark_pid_process_capability_metrics 3 2048 4096 5 2 64
./build/benchmark_pid_process_capability_metrics 3 4096 4096 3 1 64
./build/benchmark_pid_process_capability_metrics 3 2048 8192 3 1 64
```
