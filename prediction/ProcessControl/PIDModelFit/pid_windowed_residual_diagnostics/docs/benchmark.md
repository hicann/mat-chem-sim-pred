# PidWindowedResidualDiagnostics Benchmark Note

## 本机原型验证

环境：

- OS：Windows
- Python：3.11
- 数据类型：float32

测试命令：

```bash
python -m pytest prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics/tests/test_pid_windowed_residual_diagnostics.py -q
python prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics/tests/benchmark_pid_windowed_residual_diagnostics.py
```

结果：

```text
7 passed
```

## CPU Reference 性能

| B | N | windows | window | stride | lag | work items | loop ms | vectorized ms | vectorized speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 2048 | 15 | 256 | 128 | 16 | 3.93M | 176.546 | 18.141 | 9.73x |
| 128 | 4096 | 15 | 512 | 256 | 32 | 31.46M | 614.194 | 117.308 | 5.24x |
| 256 | 4096 | 15 | 512 | 256 | 32 | 62.91M | 1219.490 | 242.392 | 5.03x |
| 512 | 8192 | 15 | 1024 | 512 | 64 | 503.32M | 4660.783 | 1804.271 | 2.58x |

## 初步判断

该方向值得继续做 Ascend C 原型，但不应直接复用 Python 的 sliding-window materialization 思路。NPU 版本应按 `(batch, window)` 切分任务，在 kernel 内按窗口读取 `actual/predicted` 并融合统计量、lag 自相关和 Ljung-Box 归约，避免显式展开 `[B, W, window_size]` 中间结果。

相比 `PidTuningRuleBatch` 和标量 rollout，该方向有三个优势：

- 工作量随 `B * W * window_size * max_lag` 增长，具备足够吞吐空间。
- 输出仅为 `metrics[B,W,8]` 和 `autocorr[B,W,L]`，D2H 压力可控。
- 已有 `PidResidualDiagnostics` 在 node202 上证明同类归约/相关性扫描能获得 e2e 加速。

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
cd prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
```

构建结果：

- `libpid_windowed_residual_diagnostics_kernel_lib.so`
- `libpid_windowed_residual_diagnostics_host.so`
- `test_aclnn_pid_windowed_residual_diagnostics`
- `benchmark_pid_windowed_residual_diagnostics`

Smoke：

```text
PidWindowedResidualDiagnostics smoke windows=2 w0_mean=0 w0_mae=0.5 w0_rmse=0.707107 w0_dw=1.5 w0_autocorr=[0, -0.5]
PASSED
```

Benchmark 命令：

```bash
./build/benchmark_pid_windowed_residual_diagnostics 3 64 2048 256 128 16 5 64
./build/benchmark_pid_windowed_residual_diagnostics 3 128 4096 512 256 32 3 64
```

结果：

```text
B=64 N=2048 windows=15 window=256 stride=128 lag=16
cpu_64T_ms=4.69239
npu_kernel_ms=0.0470184
npu_resident_e2e_ms=0.0727348
kernel_speedup=99.7991
resident_e2e_speedup=64.5137
metric_max_abs=7.62939e-06
metric_max_rel=1.19104e-07
autocorr_max_abs=0
autocorr_max_rel=0

B=128 N=4096 windows=15 window=512 stride=256 lag=32
cpu_64T_ms=6.07111
npu_kernel_ms=0.0818743
npu_resident_e2e_ms=0.195025
kernel_speedup=74.1515
resident_e2e_speedup=31.1298
metric_max_abs=7.62939e-06
metric_max_rel=1.19131e-07
autocorr_max_abs=0
autocorr_max_rel=0
```

说明：

- 当前 `resident_e2e` 口径假设 `actual/predicted` 已在 Device，仅统计 kernel 和 `metrics/autocorr` D2H。
- 冷启动口径尚未统计；若输入来自 Host，完整 `actual/predicted` H2D 会影响端到端收益。
- 极小窗口不是当前性能判断口径，后续若要支持小窗口巡检，应单独补充数值稳定性和调度开销测试。

## Ascend C 原型复现命令

当前目录已补齐 host、kernel、ACL smoke 和 ACL benchmark 入口。后续在 node202 上可按以下命令验证：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
export LD_LIBRARY_PATH="$PWD/build:$PWD/build/lib:/usr/local/Ascend/ascend-toolkit/latest/lib64:${LD_LIBRARY_PATH:-}"

./build/test_aclnn_pid_windowed_residual_diagnostics 3
./build/benchmark_pid_windowed_residual_diagnostics 3 128 4096 512 256 32 5 64
./build/benchmark_pid_windowed_residual_diagnostics 3 256 4096 512 256 32 3 64
./build/benchmark_pid_windowed_residual_diagnostics 3 512 8192 1024 512 64 2 64
```

benchmark 输出字段：

- `cpu_64T_ms`：CPU 多线程 reference。
- `npu_kernel_ms`：输入已常驻 Device，仅统计 launch + stream sync。
- `npu_resident_e2e_ms`：输入已常驻 Device，统计 compute + 输出 D2H。
