# PidSopdtBatchRolloutScore

## Overview

`PidSopdtBatchRolloutScore` is an independent SOPDT batch closed-loop rollout operator for PID candidate scoring.
It evaluates a batch of SOPDT process models against a shared PID candidate set on NPU and returns the best
candidate per loop.

This operator is used in the tuning stage as the SOPDT candidate simulation and selection kernel:

- input is process model parameters plus PID candidate arrays
- output is per-loop best score and best PID gains
- implementation is independent from the earlier model-fit operators

The plant dynamics are a second-order process plus dead time:

```text
y[k+1] = a1 * y[k] + a2 * y[k-1] + b * u[k - delay]
```

Versus the FOPDT rollout this adds one history state `y[k-1]` and one coefficient (`a1`, `a2` instead of a
single `a`). The PID control law, scoring, candidate-axis SIMD, delay ring and tiling are identical to
`PidFopdtBatchRolloutScore`.

## 工程语义

SOPDT rollout 与 FOPDT/IPDT rollout 的候选评估框架相同：输入已辨识出的模型参数和一批 PID 候选，kernel 内部闭环递推，累计 `IAE/ISE/overshoot/settling_time/control_energy`，计算固定 score 并选出每条回路的最优候选。

区别在被控对象动态。SOPDT 多了一个历史输出状态：

```text
y[k+1] = a1 * y[k] + a2 * y[k-1] + b * u[k - delay]
```

它能表达二阶惯性、较慢过渡和更复杂的动态形态。相比 FOPDT，SOPDT 需要输入 `a1/a2/b/delay`，并在 kernel 中维护 `y[k]` 与 `y[k-1]` 两个输出历史。

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
| `a1` | float32 | `[B]` | first output-history coefficient (`y[k]` term) |
| `a2` | float32 | `[B]` | second output-history coefficient (`y[k-1]` term) |
| `b` | float32 | `[B]` | input gain term (`u[k-delay]` term) |
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
cd prediction/ProcessControl/PIDModelFit/pid_sopdt_batch_rollout_score
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
python tests/test_pid_sopdt_batch_rollout_score.py
```

NPU smoke:

```bash
./build/test_aclnn_pid_sopdt_batch_rollout_score 0
```

NPU / CPU benchmark:

```bash
./build/benchmark_pid_sopdt_batch_rollout_score 0 64 1024 1024 0 2 0 64   # candidate_tile=0 => auto (min(C, kLane=768))
```

## Documents

- [Algorithm](docs/algorithm.md)
- [API Reference](docs/api_reference.md)
- [Benchmark Report](docs/benchmark.md)
