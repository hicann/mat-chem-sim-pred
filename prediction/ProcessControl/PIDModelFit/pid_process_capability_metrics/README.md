# PidProcessCapabilityMetrics

## 功能说明

`PidProcessCapabilityMetrics` 面向工业过程控制中的过程能力评价和控制性能巡检场景。算子对多条回路的历史窗口数据进行批量统计，一次输出均值、标准差、Cp、Cpk、Pp、Ppk、越限数量和越限比例等指标。

该算子适合如下批量场景：

- 上百到上千个控制回路的周期性健康度巡检。
- 多窗口滚动计算 Cpk/Ppk，用于质量能力趋势分析。
- PID 整定结果上线前的历史数据质量评价。

## 与控制性能算子的区别

`PidProcessCapabilityMetrics` 是质量规格视角，只看 `values` 是否稳定落在 `[lsl, usl]` 内，不需要设定值 `sp`。`PidControlPerformanceMetrics` 是控制跟踪视角，需要 `pv/sp`，还会计算 IAE/ISE、超调、稳定时间和 Harris 指数。

```text
本算子回答：过程是否达标、波动是否过大、均值是否偏规格边缘？
控制性能算子回答：控制器是否跟得上设定值、误差和超调是否可接受？
```

例如 `pv=[9,11,9,11]`、`lsl=8`、`usl=12` 时，过程能力可能认为没有越限；但如果 `sp=[10,10,10,10]`，控制性能还会指出该回路一直围绕设定值振荡，IAE/ISE 不为 0。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `values` | float32 | `[B, N]` | 每条回路的历史采样窗口 |
| `lsl` | float32 | `[B]` | 每条回路下规格限 |
| `usl` | float32 | `[B]` | 每条回路上规格限 |
| `metrics` | float32 | `[B, 13]` | 输出指标矩阵 |

`metrics` 的最后一维顺序：

```text
0 mean
1 std_sample
2 std_population
3 cp
4 cpu
5 cpl
6 cpk
7 pp
8 ppk
9 out_of_spec_ratio
10 out_of_spec_count
11 min_value
12 max_value
```

## 指标公式

```text
Cpu = (USL - mean) / (3 * std_sample)
Cpl = (mean - LSL) / (3 * std_sample)
Cp  = (USL - LSL) / (6 * std_sample)
Cpk = min(Cpu, Cpl)
Pp  = (USL - LSL) / (6 * std_population)
Ppk = min((USL - mean)/(3*std_population), (mean - LSL)/(3*std_population))
```

## 指标说明

| 指标 | 含义 |
|------|------|
| `mean` | 窗口均值，表示过程中心位置 |
| `std_sample` | 样本标准差，分母 `N-1`，用于 `Cp/Cpk` |
| `std_population` | 总体标准差，分母 `N`，用于 `Pp/Ppk` |
| `cp` | 规格宽度相对短期波动的能力，不考虑均值偏移 |
| `cpu/cpl` | 均值到上/下规格限的裕度 |
| `cpk` | `min(cpu,cpl)`，同时考虑波动和均值偏移 |
| `pp/ppk` | 使用总体标准差的长期表现类能力指标 |
| `out_of_spec_ratio/count` | 超出 `[lsl,usl]` 的比例和数量 |
| `min_value/max_value` | 窗口内最小值和最大值 |

## 小例子

假设一条回路：

```text
values = [9, 10, 11, 10, 12]
LSL = 8
USL = 14
```

均值：

```text
mean = 10.4
```

中心化平方和：

```text
[-1.4, -0.4, 0.6, -0.4, 1.6]^2 -> m2 = 5.2
```

标准差：

```text
std_population = sqrt(5.2 / 5) = 1.0198
std_sample = sqrt(5.2 / 4) = 1.1402
```

规格宽度为 `6`：

```text
Cp = 6 / (6 * 1.1402) = 0.877
Cpu = (14 - 10.4) / (3 * 1.1402) = 1.052
Cpl = (10.4 - 8) / (3 * 1.1402) = 0.702
Cpk = min(1.052, 0.702) = 0.702
Pp = 6 / (6 * 1.0198) = 0.981
Ppk = min((14 - 10.4)/(3*1.0198), (10.4 - 8)/(3*1.0198)) = 0.784
```

所有采样点都在 `[8,14]` 内：

```text
out_of_spec_count = 0
out_of_spec_ratio = 0
min_value = 9
max_value = 12
```

## 构建

```bash
cd prediction/ProcessControl/PIDModelFit/pid_process_capability_metrics
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_process_capability_metrics.py
python tests/benchmark_pid_process_capability_metrics.py
```

NPU smoke：

```bash
./build/test_aclnn_pid_process_capability_metrics 3
```

NPU/CPU benchmark：

```bash
./build/benchmark_pid_process_capability_metrics 3 128 4096 20 3 64
```

## 文档

- [算法说明](docs/algorithm.md)
- [API 说明](docs/api_reference.md)
- [测试报告](docs/benchmark.md)
