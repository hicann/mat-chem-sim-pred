# PidStepResponseFeatures

## 功能说明

`PidStepResponseFeatures` 是 PID 候选轨迹的批量特征提取原型。它假设上游已经得到候选过程变量轨迹：

```text
pv_candidates[B, C, N], sp[B, N]
    -> features[B, C, 12]
```

该方向不做闭环递推，只做 device-side 后处理，适合与 rollout 或仿真流水线拼接。

## 与 batch rollout 的关系

当前 `pid_fopdt_batch_rollout_score`、`pid_ipdt_batch_rollout_score`、`pid_sopdt_batch_rollout_score` 已经在闭环仿真过程中融合了部分特征累计、固定评分和 best reduce：

```text
模型参数 + PID 候选
  -> 闭环仿真
  -> IAE/ISE/overshoot/settling_time/score
  -> best_result/best_idx
```

因此如果主链直接使用 batch rollout 输出最优 PID，通常不需要再调用本算子。本算子适用于另一条模块化链路：上游已经生成完整候选轨迹 `pv_candidates[B,C,N]`，并且希望保留每个候选的 12 维可解释特征，再由 host 侧策略或后续评分算子做可配置评分与选优。

两条路线的区别：

```text
batch_rollout_score:
    仿真 + 固定特征 + 固定评分 + 选优，融合高效，只输出 best

step_response_features + downstream scorer:
    不负责仿真，处理已有轨迹，保留每个候选特征，更灵活但链路更长
```

## 特征

```text
initial_value, final_value, final_abs_error,
peak_value, trough_value,
overshoot_ratio, undershoot_ratio,
rise_time, peak_time, settling_time,
iae, ise
```

## 计算说明与小例子

对单条候选轨迹：

```text
target = sp[b, N-1]
initial = pv_candidates[b,c,0]
final = pv_candidates[b,c,N-1]
delta = target - initial
```

算子扫描整条 `pv`，得到峰值、谷值、首次到达 10%/90% 的时间、最后一次跑出稳定带的时间，以及误差积分：

```text
IAE = sum(abs(sp - pv)) * sample_interval
ISE = sum((sp - pv)^2) * sample_interval
```

例如：

```text
sp = [10, 10, 10, 10, 10, 10]
pv_A = [0, 4, 8, 10, 10, 10]
pv_B = [0, 8, 12, 11, 10, 10]
```

候选 A 没有超调，但前期跟踪较慢：

```text
IAE = 10 + 6 + 2 + 0 + 0 + 0 = 18
ISE = 100 + 36 + 4 = 140
```

候选 B 跟踪更快，但存在超调：

```text
peak_value = 12
overshoot_ratio = (12 - 10) / (10 - 0) = 0.2
IAE = 10 + 2 + 2 + 1 + 0 + 0 = 15
ISE = 100 + 4 + 4 + 1 = 109
```

后续评分器可以根据业务权重决定更偏好“快”还是“稳”。

## 当前定位

该目录已从 Python reference 探索池推进为 Ascend C 原型算子，包含：

- `aclnnPidStepResponseFeatures` host API。
- `pid_step_response_features_kernel` device kernel。
- ACLNN smoke 示例。
- ACLNN benchmark。
- Python reference、测试和本机 benchmark。

该算子按 `(batch,candidate)` 切分任务，在 kernel 内融合峰值、误差积分、rise/settling time 扫描，不显式输出或回传中间轨迹特征。

## 已验证结果

- 本地 Python reference：`4 passed`。
- node202 / Ascend910B3 CANN 构建通过。
- ACLNN smoke 通过：known case 输出 `initial=0 final=10 peak=11 overshoot=0.1 rise_time=2 peak_time=4 settling_time=5 iae=15 ise=91`。
- node202 benchmark 中等和较大规模均与 CPU reference 完全对齐，`feature_max_abs=0`。

详细材料：

- [算法说明](docs/algorithm.md)
- [API Reference](docs/api_reference.md)
- [Benchmark Note](docs/benchmark.md)
- [交付说明](docs/delivery.md)

## 本机验证

```bash
python -m pytest prediction/ProcessControl/PIDModelFit/pid_step_response_features/tests/test_pid_step_response_features.py -q
python prediction/ProcessControl/PIDModelFit/pid_step_response_features/tests/benchmark_pid_step_response_features.py
```

## Ascend C 构建与 smoke

```bash
cd prediction/ProcessControl/PIDModelFit/pid_step_response_features
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
./build/test_aclnn_pid_step_response_features 0
./build/benchmark_pid_step_response_features 0 64 32 1024 5 64
```
