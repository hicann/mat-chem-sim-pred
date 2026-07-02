# PidControlPerformanceMetrics 算法说明

## 背景

控制性能巡检通常会同时计算多类指标：过程能力 Cp/Cpk/Pp/Ppk、Harris 指数、IAE/ISE/ITAE、越限比例、超调、稳定时间等。如果在 CPU 上逐指标扫描窗口，会对同一批 `pv/sp` 数据重复读取多次。

该算子将这些指标融合到一次 NPU kernel 扫描中，更适合多回路、多窗口、上游数据已位于 NPU 的整定评分流水线。

## 指标

对每条回路 `b` 的窗口 `pv[b, :]`、`sp[b, :]` 一次扫描得到：

```text
mean_pv
m2_pv
IAE = sum(abs(sp - pv)) * dt
ISE = sum((sp - pv)^2) * dt
ITAE = sum(t * abs(sp - pv)) * dt
out_of_spec_count
max_abs_error
max_positive_deviation
max_negative_deviation
last_unsettled_time
```

均值和方差采用 Welford 在线算法：

```text
delta = pv_i - mean
mean = mean + delta / count
m2 = m2 + delta * (pv_i - mean)
```

Harris 指数计算为：

```text
harris_index = clamp(mv_variance / variance_pv, 0, 1)
```

其中 `mv_variance` 由上游 MVC/最小方差基准模型估计后输入，本算子负责批量融合统计，不在 kernel 内拟合 MVC 模型。

## 适用场景

- 多回路控制性能周期性巡检。
- NPU 侧模型辨识/仿真后的评分指标融合。
- 多窗口滚动评价，减少重复扫描和 Host/Device 往返。

如果只从 CPU 传入一小段数据并单独计算一次指标，H2D/D2H 可能抵消 NPU kernel 收益。
