# PidWindowedResidualDiagnostics Algorithm

## 功能

`PidWindowedResidualDiagnostics` 对 `actual[B, N]` 与 `predicted[B, N]` 做滑动窗口残差诊断，用于发现全局残差指标掩盖的分段漂移、局部振荡和模型失配。

输出布局：

```text
metrics[B, W, 8]
autocorr[B, W, max_lag]
W = 1 + (N - window_size) / stride
```

## 指标

每个 `(batch, window)` 独立计算 8 个指标：

```text
mean_residual
std_residual
mae
rmse
max_abs_residual
fit_percent
durbin_watson
ljung_box_q
```

`autocorr[b, w, lag - 1]` 为窗口内残差在指定 lag 下的归一化自相关。

## Kernel 策略

Kernel 将每个 `(batch, window)` 映射为一个 task，并按 task 分配到多个 AI Core：

```text
task = b * W + w
window_start = w * stride
input_base = b * N + window_start
```

每个 task 在 kernel 内直接扫描原始 `actual/predicted` 窗口，不显式展开 `[B, W, window_size]` 中间张量。这样避免滑窗物化造成额外 HBM 占用，同时只回传小规模诊断结果。

## 计算过程

对窗口残差 `r_i = actual_i - predicted_i`：

```text
mean = avg(r)
centered = r - mean
sse = sum(r_i^2)
residual_energy = sum(centered_i^2)
actual_energy = sum((actual_i - avg(actual))^2)
```

再得到：

```text
std = sqrt(residual_energy / max(window_size - 1, 1))
mae = avg(abs(r))
rmse = sqrt(sse / window_size)
fit_percent = 100 * (1 - sqrt(sse) / sqrt(max(actual_energy, eps)))
durbin_watson = sum((r_i - r_{i-1})^2) / max(sse, eps)
```

多 lag 自相关与 Ljung-Box 风格统计量：

```text
autocorr_lag = sum(centered_i * centered_{i-lag}) / max(residual_energy, eps)
ljung_box_q = n * (n + 2) * sum(autocorr_lag^2 / max(n - lag, 1))
```

## 复杂度

每个窗口的主计算复杂度为：

```text
O(window_size * max_lag)
```

整体工作量约为：

```text
B * W * window_size * max_lag
```

输出规模为：

```text
B * W * (8 + max_lag)
```

## 当前限制

- 当前 kernel 是 GlobalTensor 标量访问原型，尚未做 UB 分块和向量化。
- 当前验证聚焦中大窗口场景；极小窗口会被 launch 和调度开销主导，应单独评估。
- `resident_e2e` 口径假设 `actual/predicted` 已在 Device；若输入来自 Host，还需补 cold H2D 端到端数据。
- 若后续继续优化，优先优化 lag 维度的 UB staging，减少重复 Global Memory 读取。
