# PidProcessCapabilityMetrics 算法说明

## 背景

Cp/Cpk/Pp/Ppk 是工业质量和过程能力评价的常见指标。在控制性能巡检场景中，通常需要对上百到上千个回路的历史窗口批量计算这些指标。CPU 实现若按回路逐个扫描窗口，会出现大量重复循环和内存访问。

## 算法

对每条回路 `b` 的窗口 `values[b, :]`，算子一次扫描得到：

```text
mean
m2
min_value
max_value
out_of_spec_count
```

其中均值和方差采用 Welford 在线算法更新，减少 `sum_sq / N - mean^2` 在均值较大、方差较小时的浮点抵消：

```text
count = i + 1
delta = x_i - mean
mean = mean + delta / count
m2 = m2 + delta * (x_i - mean)
```

再计算：

```text
std_population = sqrt(m2 / N)
std_sample = sqrt(m2 / (N - 1))
Cp = (USL - LSL) / (6 * std_sample)
Cpk = min((USL - mean)/(3*std_sample), (mean - LSL)/(3*std_sample))
Pp = (USL - LSL) / (6 * std_population)
Ppk = min((USL - mean)/(3*std_population), (mean - LSL)/(3*std_population))
```

## NPU 实现

- 输入 `values[B, N]` 按回路连续存储。
- Host 侧将 `batch/sample_count/core_num` 写入 workspace tiling。
- Kernel 侧按回路切分到 AI Core，每个 AI Core 处理一段 `B`。
- 每条回路只读取一次窗口数据，融合输出 13 个指标。

## 适用规模

该算子更适合：

- `B >= 100`
- `N >= 1000`
- 多窗口或周期性巡检重复运行

若只有少量回路或很短窗口，Host 调度和 H2D/D2H 拷贝可能抵消 NPU 计算收益。
