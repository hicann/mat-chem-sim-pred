# PidTuningRuleBatch 算法说明

## 原理

针对 FOPDT 模型参数——过程增益 `K`、时间常数 `T`、纯滞后 `L`，以及 IMC 的闭环时间常数 `lambda`——对每条回路一次性输出三类经典整定规则的 PID 参数：

```text
Ziegler-Nichols reaction curve
IMC PID
Cohen-Coon PID
```

三组规则均为 FOPDT 参数到 `(Kp, Ki, Kd)` 的闭式映射，逐 batch 并行计算。

## 输出

```text
pid_params[B, 3, 3]   # 规则维顺序 (ziegler_nichols, imc, cohen_coon)，最后一维 (Kp, Ki, Kd)
diagnostics[B, 3, 4]  # 最后一维 (valid, dead_time_ratio, aggressiveness, lambda_ratio)
```

- `valid`：该规则在当前 FOPDT 参数下是否产出有效（有限、非负、滞后比在适用范围内）参数。
- `dead_time_ratio = L / T`：滞后主导程度，过大时部分规则不再适用。
- `aggressiveness`：整定激进程度的相对指标。
- `lambda_ratio = lambda / T`：IMC 整定的相对带宽。

## 适用场景

该算子面向大规模参数网格 / 数字孪生中批量回路的 PID 初值生成，也用于 FOPDT E2E 链路中的候选生成阶段。由于是逐条型的轻量闭式计算、算术强度低，单独调用时单卡 NPU 难以摊薄 H2D/D2H 与 launch 开销；推荐把它作为 device-resident 整定流水线的一环使用，而不是作为独立性能卖点。

## 精度

Python 参考实现 `tuning_rule_batch` 与逐条循环实现 `tuning_rule_batch_cpu_loop` 使用相同闭式公式；测试断言两者 `pid_params` 与 `diagnostics` 一致，并对解析可算的简单工况给出固定校验点。
