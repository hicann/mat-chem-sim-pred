# PidIpdtBatchRolloutScore

## Overview

`PidIpdtBatchRolloutScore` is an independent IPDT batch closed-loop rollout operator for PID candidate scoring.
It evaluates a batch of IPDT process models against a shared PID candidate set on NPU and returns the best
candidate per loop.

This operator is used in the tuning stage as the IPDT candidate simulation and selection kernel:

- input is process model parameters plus PID candidate arrays
- output is per-loop best score and best PID gains
- implementation is independent from the earlier model-fit operators

The plant dynamics are a pure integrator plus dead time:

```text
y[k+1] = y[k] + b * u[k - delay]
```

This is the `a = 1` (no self-regulation) special case of the FOPDT rollout, so there is no `a`
coefficient input. The PID control law, scoring, candidate-axis SIMD, delay ring and tiling are
identical to `PidFopdtBatchRolloutScore`.

## 工程语义

IPDT rollout 与 FOPDT/SOPDT rollout 的候选评估框架相同：输入已辨识出的模型参数和一批 PID 候选，kernel 内部闭环递推，累计 `IAE/ISE/overshoot/settling_time/control_energy`，计算固定 score 并选出每条回路的最优候选。

区别只在被控对象动态。IPDT 是积分对象：

```text
y[k+1] = y[k] + b * u[k - delay]
```

它没有 FOPDT 的自回归衰减项 `a*y[k]`，因此过程本身没有自稳能力，输入作用会持续累积到输出上。对于液位、库存、积分型热量累积等对象，这类模型比一阶自稳模型更贴近。

PID 候选不由本算子生成；本算子只接收 `kp[M]`、`ki[M]`、`kd[M]` 并评估。候选可以来自整定规则、规则附近扰动、人工网格或外部优化器。

当前 batch rollout 已经融合了候选特征、候选评分和候选选优；如果使用本算子直接输出 `best_result/best_idx`，通常不需要再接独立候选评分/选优算子。`pid_step_response_features` 仅用于需要保留完整候选轨迹特征表的另一条模块化链路。

Current status:

- correctness is validated against the in-process CPU reference (the `benchmark` program), max quality
  rel-err `< 1e-3`
- the rollout is a serial time recurrence; the candidate axis is evaluated with a wide vector SIMD lane
  (`kLane=768`) so the inner loop is throughput-bound rather than latency-bound

See [benchmark report](docs/benchmark.md) for the measured results.

## Inputs And Outputs

| Tensor | Dtype | Shape | Meaning |
|---|---|---|---|
| `b` | float32 | `[B]` | per-step integration gain (`y[k+1] = y[k] + b*u[k-delay]`) |
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
cd prediction/ProcessControl/PIDModelFit/pid_ipdt_batch_rollout_score
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
python tests/test_pid_ipdt_batch_rollout_score.py
```

NPU smoke:

```bash
./build/test_aclnn_pid_ipdt_batch_rollout_score 0
```

NPU / CPU benchmark:

```bash
./build/benchmark_pid_ipdt_batch_rollout_score 0 64 1024 1024 0 2 0 64   # candidate_tile=0 => auto (min(C, kLane=768))
```

## Documents

- [Algorithm](docs/algorithm.md)
- [API Reference](docs/api_reference.md)
- [Benchmark Report](docs/benchmark.md)
