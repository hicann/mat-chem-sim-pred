# PidFopdtBatchRolloutScore

## Overview

`PidFopdtBatchRolloutScore` is an independent FOPDT batch closed-loop rollout operator for PID candidate scoring.
It evaluates a batch of FOPDT process models against a shared PID candidate set on NPU and returns the best
candidate per loop.

This operator is used in the tuning stage as the FOPDT candidate simulation and selection kernel:

- input is process model parameters plus PID candidate arrays
- output is per-loop best score and best PID gains
- implementation is independent from the earlier model-fit operators

## 工程语义

候选仿真不是模型辨识后处理。模型辨识/残差评估使用历史 `mv/pv`，回答“模型能否解释过去的数据”；本算子在已知 FOPDT 模型后，用 PID 自己计算出的未来控制量 `u` 做闭环递推，回答“如果使用某组 PID 参数，未来闭环响应好不好”。

FOPDT 离散对象为：

```text
y[k+1] = a * y[k] + b * u[k - delay]
```

PID 控制律为：

```text
error = sp - y
integral += error * sample_interval
derivative = (error - prev_error) / sample_interval
u = Kp * error + Ki * integral + Kd * derivative
```

算子会对每个候选 `Kp/Ki/Kd` 仿真整段闭环响应，在递推过程中累计 `IAE/ISE/overshoot/settling_time/control_energy`，按固定权重形成 `score`，最后输出每条回路的最优候选。也就是说，当前主链已经把“候选特征提取、候选评分、候选选优”融合在同一个 rollout kernel 中，只输出 best，不输出所有候选的完整特征表。

PID 候选来源不由本算子决定。它只接收已经准备好的 `kp[M]`、`ki[M]`、`kd[M]` 数组；这些候选可以来自 Ziegler-Nichols/IMC/Cohen-Coon 规则，也可以来自规则结果附近的扰动网格、人工网格、历史现场参数附近搜索或外部优化器。当前 accuracy E2E 使用三套规则产生的 3 个候选；performance E2E 使用手工生成的大规模候选网格，候选数可扫到 256、512、1024、4096、16384 等。

## 与其他 rollout/候选评估算子的关系

- `pid_ipdt_batch_rollout_score` 使用积分加滞后对象：`y[k+1] = y[k] + b*u[k-delay]`。
- `pid_sopdt_batch_rollout_score` 使用二阶对象：`y[k+1] = a1*y[k] + a2*y[k-1] + b*u[k-delay]`。
- 早期 FOPDT 标量 rollout 原型已从本次正式提交中撤回；本算子是候选维向量化/tiling 后的正式版本。
- `pid_step_response_features` 适用于上游已经产生完整 `pv_candidates[B,C,N]` 且需要保留每个候选可解释特征的场景；当前 batch rollout 主链已经融合固定评分和 best 选择，不需要再接独立候选评分/选优算子。

## 小例子

只看比例控制以便手算，设：

```text
a = 1.0, b = 0.5, delay = 0, y0 = 0, sp = 1
候选 A: Kp = 1, Ki = 0, Kd = 0
候选 B: Kp = 3, Ki = 0, Kd = 0
```

候选 A：

```text
k=0: error=1,   u=1,   y_next=0 + 0.5*1   = 0.5
k=1: error=0.5, u=0.5, y_next=0.5+0.5*0.5 = 0.75
k=2: error=0.25,u=0.25,y_next=0.75+0.5*0.25 = 0.875
```

候选 B：

```text
k=0: error=1,    u=3,    y_next=1.5
k=1: error=-0.5, u=-1.5, y_next=0.75
k=2: error=0.25, u=0.75, y_next=1.125
```

候选 B 响应更快但有超调/振荡；候选 A 更稳但慢。rollout 算子会把这类轨迹差异压成 score，并选出代价最小的 PID。

Current status:

- correctness has been validated against the CPU reference on `node202 / Ascend910B3`; NPU output is bit-identical
- the rollout is a serial time recurrence; the candidate axis is evaluated with a wide vector SIMD lane
  (`kLane=768`) so the inner loop is throughput-bound rather than latency-bound
- on a single card the NPU kernel runs roughly 4-7x the 64-thread CPU baseline at the candidate counts that
  dominate the tuning sweep (see the benchmark report), with no accuracy trade-off

See [benchmark report](docs/benchmark.md) for the measured results.

## Inputs And Outputs

| Tensor | Dtype | Shape | Meaning |
|---|---|---|---|
| `a` | float32 | `[B]` | FOPDT discrete coefficient |
| `b` | float32 | `[B]` | FOPDT discrete gain term |
| `delay` | int32 | `[B]` | input delay, clamped to `0..31` in kernel |
| `y0` | float32 | `[B]` | initial output |
| `sp` | float32 | `[B]` | setpoint |
| `kp` | float32 | `[M]` | PID candidate Kp |
| `ki` | float32 | `[M]` | PID candidate Ki |
| `kd` | float32 | `[M]` | PID candidate Kd |
| `best_result` | float32 | `[B, 8]` | best candidate metrics per loop |
| `best_idx` | int32 | `[B]` | best candidate index per loop |

`best_result` layout:

```text
best_score,best_kp,best_ki,best_kd,best_iae,best_ise,best_overshoot,best_settling_time
```

## Build

```bash
cd prediction/ProcessControl/PIDModelFit/pid_fopdt_batch_rollout_score
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j$(nproc)
```

Note:

- this project now defaults to `Release` if `CMAKE_BUILD_TYPE` is not specified
- on `node202`, runtime typically needs:

```bash
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64:$PWD/build:$PWD/build/lib:${LD_LIBRARY_PATH}
```

## Test

Python reference test:

```bash
python tests/test_pid_fopdt_batch_rollout_score.py
```

NPU smoke:

```bash
./build/test_aclnn_pid_fopdt_batch_rollout_score 0
```

NPU / CPU benchmark:

```bash
./build/benchmark_pid_fopdt_batch_rollout_score 0 64 1024 1024 0 2 0 64   # candidate_tile=0 => auto (min(C, kLane=768))
```

## Documents

- [Algorithm](docs/algorithm.md)
- [API Reference](docs/api_reference.md)
- [Benchmark Report](docs/benchmark.md)
