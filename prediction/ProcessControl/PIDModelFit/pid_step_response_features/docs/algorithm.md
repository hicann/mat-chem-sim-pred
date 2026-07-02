# PidStepResponseFeatures Algorithm

## 功能

`PidStepResponseFeatures` 是 PID 候选阶跃响应轨迹的 device-side 特征提取算子。它不做闭环递推，假设上游已经生成候选过程变量轨迹：

```text
pv_candidates[B, C, N], sp[B, N]
    -> features[B, C, 12]
```

该算子用于把完整候选轨迹压缩成可用于排序、筛选或评分的 12 个标量特征，避免把大规模轨迹搬回 CPU 后再逐候选扫描。

## 输出特征

每个 `(batch, candidate)` 输出 12 个特征：

```text
0  initial_value
1  final_value
2  final_abs_error
3  peak_value
4  trough_value
5  overshoot_ratio
6  undershoot_ratio
7  rise_time
8  peak_time
9  settling_time
10 iae
11 ise
```

## 计算定义

对单条候选轨迹：

```text
target = sp[b, N - 1]
initial = pv_candidates[b, c, 0]
final = pv_candidates[b, c, N - 1]
delta = target - initial
direction = 1 if delta >= 0 else -1
abs_delta = max(abs(delta), eps)
```

峰值和谷值：

```text
peak_value = max(pv)
trough_value = min(pv)
peak_time = argmax(pv) * sample_interval
```

超调和欠调：

```text
overshoot_ratio = max(direction * (peak_value - target), 0) / abs_delta
undershoot_ratio = max(direction * (target - trough_value), 0) / abs_delta
```

上升时间：

```text
normalized[i] = direction * (pv[i] - initial) / abs_delta
rise_time = first_time(normalized >= 0.9) - first_time(normalized >= 0.1)
```

若没有跨过阈值，阈值时间回退到最后一个采样点，与 Python reference 保持一致。

调节时间与误差积分：

```text
band = max(abs_delta * settle_band_ratio, 1e-4)
settling_time = (last index with abs(sp[i] - pv[i]) > band + 1) * sample_interval
iae = sum(abs(sp[i] - pv[i])) * sample_interval
ise = sum((sp[i] - pv[i])^2) * sample_interval
```

## Kernel 切分

Kernel 将每个 `(batch, candidate)` 映射为一个 task：

```text
task = batch_index * C + candidate_index
```

每个 AI Core 处理连续 task 区间。每个 task 在 kernel 内直接扫描 `pv_candidates` 和 `sp`，融合 peak/trough、rise/settling time、IAE/ISE 等统计，不创建 `[B,C,N]` 中间临时特征数组。

## 复杂度

每个候选的主计算复杂度为：

```text
O(N)
```

整体工作量为：

```text
B * C * N
```

输出规模为：

```text
B * C * 12
```

## 当前限制

- 当前 kernel 是 GlobalTensor 标量扫描原型，尚未做 UB 分块和向量化。
- `resident_e2e` 口径假设 `pv_candidates/sp` 已在 Device；若输入来自 Host，还需要补 cold H2D 端到端数据。
- `rise_time` 和 `settling_time` 含条件判断，后续优化时需要保持与 Python reference 的阈值口径一致。
