# PidStepResponseFeatures Benchmark Note

## 目标

`PidStepResponseFeatures` 是第二个新算子方向尝试，用于对候选阶跃响应轨迹做批量后处理。它不做闭环递推，只假设上游已有：

```text
pv_candidates[B, C, N], sp[B, N]
```

并输出：

```text
features[B, C, 12]
```

## 本机复现命令

```bash
python -m pytest prediction/ProcessControl/PIDModelFit/pid_step_response_features/tests/test_pid_step_response_features.py -q
python prediction/ProcessControl/PIDModelFit/pid_step_response_features/tests/benchmark_pid_step_response_features.py
```

## 初步判断口径

本机 benchmark 只用于筛方向：

- `loop_ms`：Python loop reference，模拟逐候选逐时间扫描。
- `numpy_ms`：NumPy vectorized reference，表示该方向是否能被批量化。
- 若 `numpy_ms` 仍明显偏高，说明该方向需要谨慎上 Ascend C；若批量化收益稳定，再考虑 kernel 原型。

## 本机结果

```text
4 passed
```

| B | C | N | loop ms | numpy ms | speedup |
|---:|---:|---:|---:|---:|---:|
| 32 | 16 | 512 | 258.632 | 6.349 | 40.73x |
| 64 | 32 | 1024 | 2166.663 | 41.855 | 51.77x |
| 128 | 64 | 1024 | 8828.969 | 211.420 | 41.76x |
| 256 | 64 | 2048 | 34199.989 | 657.373 | 52.03x |

## 结论

该方向值得继续做 Ascend C 原型。它没有闭环递推依赖，计算结构主要是按 `(batch,candidate)` 扫描轨迹并融合峰值、误差积分、rise time 和 settling time，适合作为上游 rollout 或仿真输出后的 device-side 特征提取。

风险点：

- 当前 NumPy 版本会产生若干 `[B,C,N]` 临时数组，Ascend C 版本应在单 kernel 内边扫边归约。
- `rise_time` 和 `settling_time` 包含首次/末次条件判断，kernel 中需要注意分支和精度口径。

## Ascend C 原型验证

环境：

- NPU 环境：node202
- SOC：Ascend910B3
- CANN：`/usr/local/Ascend/ascend-toolkit`
- device：3
- CPU 对比线程数：64

构建命令：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_step_response_features
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
```

构建结果：

- `libpid_step_response_features_kernel_lib.so`
- `libpid_step_response_features_host.so`
- `test_aclnn_pid_step_response_features`
- `benchmark_pid_step_response_features`

Smoke：

```text
PidStepResponseFeatures smoke initial=0 final=10 peak=11 overshoot=0.1 rise_time=2 peak_time=4 settling_time=5 iae=15 ise=91
PASSED
```

Benchmark 命令：

```bash
./build/benchmark_pid_step_response_features 3 64 32 1024 5 64
./build/benchmark_pid_step_response_features 3 128 64 1024 3 64
```

结果：

```text
B=64 C=32 N=1024
cpu_64T_ms=4.41643
npu_kernel_ms=0.0547166
npu_resident_e2e_ms=0.101893
kernel_speedup=80.7146
resident_e2e_speedup=43.3438
feature_max_abs=0
feature_max_rel=0

B=128 C=64 N=1024
cpu_64T_ms=6.87986
npu_kernel_ms=0.0792977
npu_resident_e2e_ms=0.164775
kernel_speedup=86.7599
resident_e2e_speedup=41.7531
feature_max_abs=0
feature_max_rel=0
```

说明：

- 当前 `resident_e2e` 口径假设 `pv_candidates/sp` 已在 Device，仅统计 kernel 和 `features` D2H。
- 冷启动口径尚未统计；若输入来自 Host，完整候选轨迹 H2D 会影响端到端收益。
- 当前 Ascend C 原型没有物化 NumPy reference 中的中间数组，直接在 kernel 内单次扫描完成 12 个特征。
