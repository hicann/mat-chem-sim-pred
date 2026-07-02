# PidResidualDiagnostics

## 功能说明

`PidResidualDiagnostics` 面向 PID 模型辨识后的模型有效性检查。算子对 `actual[B, N]` 和 `predicted[B, N]` 计算残差基础指标、多 lag 自相关和 Ljung-Box 风格白噪声统计量：

```text
residual = actual - predicted
metrics[B, 8], autocorr[B, max_lag]
```

该方向服务于“模型是否足够可信、残差是否仍有结构性动态”的工程问题。相比闭环递推，它主要是批量归约和相关性扫描，适合作为 NPU 侧模型辨识/评分流水线的后处理。

## 残差和误差的关系

在本算子里，残差采用固定符号约定：

```text
residual[i] = actual[i] - predicted[i]
```

其中 `actual` 是实测过程输出，`predicted` 是辨识模型给出的预测输出。工程上常说的“误差”可以泛指实测值和预测值之间的差异；“残差”则特指在已经完成模型拟合之后，样本点上的 `actual - predicted`。因此残差可以看作模型辨识场景下的一种预测误差。

这个符号约定会影响 `mean_residual` 的解释：如果 `mean_residual > 0`，说明实测值整体高于预测值，模型平均预测偏低；如果 `mean_residual < 0`，说明模型平均预测偏高。`mae`、`rmse` 和 `max_abs_residual` 使用绝对值或平方，不受符号方向影响。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `actual` | float32 | `[B, N]` | 实测输出 |
| `predicted` | float32 | `[B, N]` | 模型预测输出 |
| `metrics` | float32 | `[B, 8]` | 残差诊断指标 |
| `autocorr` | float32 | `[B, max_lag]` | residual autocorrelation lag 1..max_lag |

`metrics` 顺序：

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

指标含义：

| 指标 | 含义 | 工程解释 |
|------|------|----------|
| `mean_residual` | 残差均值 | 越接近 0 越好；明显偏正/偏负表示模型存在系统偏差 |
| `std_residual` | 残差标准差 | 描述残差波动大小，越小表示预测误差越稳定 |
| `mae` | 平均绝对误差 | 平均每个采样点偏差多少，直观且对尖峰不如 RMSE 敏感 |
| `rmse` | 均方根误差 | 对大误差更敏感；若明显大于 MAE，通常说明存在尖峰误差 |
| `max_abs_residual` | 最大绝对残差 | 最坏采样点偏差，用于发现局部异常或突变 |
| `fit_percent` | 拟合优度百分比 | 越高越好；接近 100 表示模型解释了大部分实际变化，低于 0 表示比均值预测还差 |
| `durbin_watson` | 一阶残差自相关诊断 | 接近 2 通常较好；小于 2 多表示正自相关，残差连续同向，模型可能漏掉慢动态或滞后 |
| `ljung_box_q` | 多 lag 白噪声统计量 | 越小越好；过大说明残差仍有结构性相关，模型没有解释干净 |

常见使用方式是先看 `fit_percent/rmse` 判断误差总量，再看 `mean_residual` 判断偏差方向，最后结合 `durbin_watson`、`ljung_box_q` 和 `autocorr` 判断残差是否还存在可预测的动态结构。若残差诊断不通过，可以拒绝该辨识结果、扩大候选网格、切换 FOPDT/IPDT/SOPDT 模型族、重选数据窗口，或把后续 PID 参数整定降级为人工审核/保守策略。

## 当前定位

该算子是本次正式提交的模型诊断算子，用于 FOPDT/IPDT/SOPDT 模型辨识后的 device-side 残差质量检查。它与 fit 算子中的 SSE 不重复：SSE 只用于候选模型的最小二乘选优，残差诊断进一步判断误差是否有偏、是否自相关、是否接近白噪声，从而决定模型是否可直接用于后续 PID 整定。

## node202 验证结论

当前版本已在 node202 / Ascend910B3 上完成 CANN 编译、ACLNN smoke 和 benchmark。中大规模下 kernel-only 与 e2e 均相对 CPU 64 线程有稳定收益，适合作为 NPU resident 模型辨识流水线的后处理。

代表性结果：

| B | N | max_lag | CPU 64T ms | NPU kernel ms | NPU e2e ms | e2e/CPU64T |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1024 | 16 | 0.468949 | 0.092059 | 0.442651 | 1.06x |
| 256 | 1024 | 32 | 1.39999 | 0.0552526 | 0.359782 | 3.89x |
| 512 | 2048 | 32 | 2.33311 | 0.075771 | 0.957573 | 2.44x |
| 1024 | 2048 | 64 | 8.62803 | 0.0683055 | 1.85133 | 4.66x |

详细记录见 [benchmark 报告](docs/benchmark.md)。

## 文档

- [算法说明](docs/algorithm.md)
- [API 说明](docs/api_reference.md)
- [benchmark 报告](docs/benchmark.md)
