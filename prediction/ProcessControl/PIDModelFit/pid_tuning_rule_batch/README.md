# PidTuningRuleBatch

## 功能说明

`PidTuningRuleBatch` 面向 FOPDT 模型参数的批量 PID 经验整定。输入过程增益 `K`、时间常数 `T`、纯滞后 `L` 和 IMC 闭环时间常数 `lambda`，一次输出三类规则的 PID 参数：

```text
Ziegler-Nichols reaction curve
IMC PID
Cohen-Coon PID
```

输出：

```text
pid_params[B, 3, 3]      # Kp, Ki, Kd
diagnostics[B, 3, 4]     # valid, dead_time_ratio, aggressiveness, lambda_ratio
```

该算子用于大规模模型参数网格、数字孪生批量回路和 PID 初值生成场景。

## 输入输出含义

输入来自上游 FOPDT 模型辨识结果：

```text
process_gain[B]   # K，过程增益
time_constant[B]  # T，一阶时间常数
dead_time[B]      # L，纯滞后
lambda_value[B]   # lambda，IMC 闭环时间常数
```

输出 `pid_params[B, 3, 3]` 的第二维是规则，顺序为：

```text
0 Ziegler-Nichols reaction curve
1 IMC PID
2 Cohen-Coon PID
```

最后一维为：

```text
0 Kp
1 Ki
2 Kd
```

输出 `diagnostics[B, 3, 4]` 的最后一维为：

```text
0 valid              # K/T/L/lambda 是否有效
1 dead_time_ratio    # L / T
2 aggressiveness     # |Kp| + T*|Ki| + |Kd|/T
3 lambda_ratio       # lambda / L
```

## 与 PID 候选仿真的关系

该算子只负责从 FOPDT 模型参数生成三套经验整定 PID 初值。它不会做闭环仿真，也不会判断哪套 PID 最好。后续若接 `pid_fopdt_batch_rollout_score`，可以把这三套规则作为候选：

```text
K,T,L -> tuning_rule_batch -> 3 组 Kp/Ki/Kd -> rollout_score -> best
```

当前 accuracy E2E 就是这种口径，每条回路只在 ZN/IMC/Cohen-Coon 三个候选中选最优。性能 E2E 为了压测 rollout，则使用手工生成的大规模 `kp/ki/kd` 网格，候选数可扫到 256、512、1024、4096、16384 等；这些大规模候选不是 `tuning_rule_batch` 直接产生的。

工程上更常见的用法是把三套规则作为种子，再在每个种子附近做比例扰动或网格扩展，形成几十到几千个 PID 候选后再仿真筛选。

## 计算公式

记：

```text
tau = T
theta = L
lam = lambda
ratio = theta / tau
```

Ziegler-Nichols：

```text
Kp = 1.2 * tau / (K * theta)
Ti = 2 * theta
Td = 0.5 * theta
Ki = Kp / Ti
Kd = Kp * Td
```

IMC PID：

```text
Kp = (tau + 0.5 * theta) / (K * (lam + 0.5 * theta))
Ti = tau + 0.5 * theta
Td = tau * theta / (2 * tau + theta)
Ki = Kp / Ti
Kd = Kp * Td
```

Cohen-Coon：

```text
Kp = tau / (K * theta) * (4/3 + theta / (4 * tau))
Ti = theta * (32 + 6 * ratio) / (13 + 8 * ratio)
Td = theta * 4 / (11 + 2 * ratio)
Ki = Kp / Ti
Kd = Kp * Td
```

## 手推例子

假设两条回路：

```text
process_gain  = [2, 0]
time_constant = [10, 8]
dead_time     = [2, 1]
lambda_value  = [4, 2]
```

第 0 条回路有效：

```text
K = 2, T = 10, L = 2, lambda = 4
ratio = 0.2
lambda_ratio = 2
```

Ziegler-Nichols：

```text
Kp = 1.2 * 10 / (2 * 2) = 3
Ti = 2 * 2 = 4
Td = 0.5 * 2 = 1
Ki = 3 / 4 = 0.75
Kd = 3 * 1 = 3
```

IMC：

```text
Kp = (10 + 1) / (2 * (4 + 1)) = 1.1
Ti = 11
Td = 10 * 2 / (20 + 2) = 0.9091
Ki = 1.1 / 11 = 0.1
Kd = 1.1 * 0.9091 = 1.0
```

Cohen-Coon：

```text
Kp = 10 / (2 * 2) * (4/3 + 2/40) = 3.4583
Ti = 2 * (32 + 6*0.2) / (13 + 8*0.2) = 4.5479
Td = 2 * 4 / (11 + 2*0.2) = 0.7018
Ki = 3.4583 / 4.5479 = 0.7605
Kd = 3.4583 * 0.7018 = 2.4276
```

因此：

```text
pid_params[0] =
[
  [3.0000, 0.7500, 3.0000],
  [1.1000, 0.1000, 1.0000],
  [3.4583, 0.7605, 2.4276],
]
```

第 1 条回路 `K=0`，输入无效，三套 PID 参数清零，`valid=0`。

## 当前结论

当前版本已经在 node202 / Ascend910B3 上完成 smoke、构建和 benchmark 验证，并作为 FOPDT E2E 链路中的候选生成阶段保留。

需要注意的是，该算子每条回路只做少量闭式公式，算术强度低，不适合作为单独性能卖点。它的工程价值主要是让 `model parameters -> PID candidates -> rollout score` 这条链路可以保持 device-resident；真正的 NPU 加速主力仍是 basis-GEMM 模型辨识、批量闭环 rollout、残差诊断和指标类融合后处理。
