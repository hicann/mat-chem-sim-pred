# PidResidualDiagnostics 算法说明

## 背景

PID 模型辨识完成后，不能只看最小 SSE 或最优参数，还需要判断模型残差是否接近白噪声。如果残差仍存在明显自相关，说明模型没有解释掉过程动态，后续 PID 参数整定可能基于不可靠模型。

`PidResidualDiagnostics` 面向多回路模型验证场景，对 `actual[B, N]` 和 `predicted[B, N]` 批量计算残差质量指标和多 lag 自相关：

```text
residual[b, i] = actual[b, i] - predicted[b, i]
metrics[B, 8], autocorr[B, max_lag]
```

该算子适合作为 FOPDT/IPDT/SOPDT 模型辨识或 NPU 侧预测流水线后的后处理步骤。

## 指标

对每条回路 `b`，先计算残差均值和中心化残差：

```text
mean_residual = mean(residual)
centered_residual[i] = residual[i] - mean_residual
residual_energy = sum(centered_residual[i]^2)
sse = sum(residual[i]^2)
```

输出指标顺序：

```text
0 mean_residual
1 std_residual
2 mae
3 rmse
4 max_abs_residual
5 fit_percent
6 durbin_watson
7 ljung_box_q
```

其中：

```text
std_residual = sqrt(residual_energy / (N - 1))
mae = mean(abs(residual))
rmse = sqrt(sse / N)
fit_percent = 100 * (1 - sqrt(sse) / sqrt(sum((actual - mean(actual))^2)))
durbin_watson = sum((residual[i] - residual[i - 1])^2) / sse
```

多 lag 自相关：

```text
autocorr[lag] =
    sum((residual[i] - mean_residual) * (residual[i - lag] - mean_residual))
    / residual_energy
```

Ljung-Box 风格统计量：

```text
ljung_box_q = N * (N + 2) * sum(autocorr[lag]^2 / (N - lag))
```

## NPU 映射

当前第一阶段原型采用每条回路一个逻辑工作单元：

```text
block -> one or more batch rows
for each row:
    scan actual/predicted once for means and residual energy
    scan once for scalar residual metrics
    scan per lag for autocorr and Q
```

该实现的主要计算量为 `B * N * max_lag`，输出只有 `B * (8 + max_lag)` 个 float，适合上游数据已在 Device 的模型验证流水线。

## 输出取舍

当前版本固定输出 8 个残差诊断指标，并使用 `fit_percent` 作为拟合优度指标。`fit_percent` 与工业 PID 工具中常见的模型拟合百分比一致，便于直接判断模型是否解释了主要过程变化。

## 适用场景

- PID 模型辨识后的残差白噪声检查。
- 大批量控制回路模型质量巡检。
- NPU 侧 FOPDT/IPDT/SOPDT fit pipeline 后处理。
- 对比多个建模方法时的残差结构化诊断。
