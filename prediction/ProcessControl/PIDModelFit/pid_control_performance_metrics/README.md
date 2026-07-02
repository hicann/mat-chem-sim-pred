# PidControlPerformanceMetrics

## 功能说明

`PidControlPerformanceMetrics` 面向多回路控制性能巡检和 NPU 侧整定评分流水线。算子对 `pv[B, N]` 和 `sp[B, N]` 进行一次批量扫描，同时输出过程能力、Harris 指数、误差积分、越限、超调、稳定时间等 20 个指标。

相比单独计算 Cpk、IAE、ISE、超调等多个小指标，该算子将多项统计融合到一次窗口扫描中，更适合作为 NPU 上游仿真/模型评分后的后处理步骤。

## 与过程能力算子的区别

`PidProcessCapabilityMetrics` 只需要 `values/lsl/usl`，核心回答“过程数据是否稳定落在规格限内”。本算子额外需要 `sp` 和 `mv_variance`，核心回答“控制器是否把 `pv` 跟踪到 `sp`，控制效果好不好”。

```text
process_capability_metrics:
    values[B,N], lsl[B], usl[B]
    -> 均值、标准差、Cp/Cpk/Pp/Ppk、越限、min/max

control_performance_metrics:
    pv[B,N], sp[B,N], lsl[B], usl[B], mv_variance[B]
    -> 过程能力 + Harris + 跟踪误差 + 超调/欠调 + 稳定时间
```

因此两者有部分指标重叠，但使用视角不同：前者是质量规格视角，后者是控制跟踪视角。如果只关心规格限能力，用过程能力算子更轻；如果要评价 PID 控制效果，用本算子。

## 输入输出

| 名称 | 类型 | Shape | 说明 |
|------|------|-------|------|
| `pv` | float32 | `[B, N]` | 过程变量历史窗口 |
| `sp` | float32 | `[B, N]` | 设定值历史窗口 |
| `lsl` | float32 | `[B]` | 下规格限 |
| `usl` | float32 | `[B]` | 上规格限 |
| `mv_variance` | float32 | `[B]` | Harris 指数使用的最小方差/MVC 基准方差 |
| `metrics` | float32 | `[B, 20]` | 输出指标矩阵 |

`metrics` 的最后一维顺序：

```text
0 mean_pv
1 std_pv_sample
2 std_pv_population
3 cp
4 cpk
5 pp
6 ppk
7 harris_index
8 iae
9 ise
10 itae
11 mae
12 rmse
13 max_abs_error
14 out_of_spec_ratio
15 out_of_spec_count
16 overshoot_ratio
17 undershoot_ratio
18 settling_time
19 final_abs_error
```

## 指标说明

| 指标 | 含义 |
|------|------|
| `mean_pv` | `pv` 均值，表示过程中心位置 |
| `std_pv_sample` / `std_pv_population` | 样本/总体标准差，表示过程波动 |
| `cp/cpk/pp/ppk` | 规格限能力指标，衡量过程波动和均值偏移是否满足上下限 |
| `harris_index` | Harris 指数，当前实现为 `clamp(mv_variance / variance_pv, 0, 1)`，`mv_variance` 由上游 MVC/最小方差基准输入 |
| `iae/ise/itae` | 绝对误差、平方误差、时间加权绝对误差积分 |
| `mae/rmse/max_abs_error` | 平均绝对误差、均方根误差、最大绝对误差 |
| `out_of_spec_ratio/count` | `pv` 超出 `[lsl, usl]` 的比例和数量 |
| `overshoot_ratio/undershoot_ratio` | 相对规格宽度的最大正/负偏差 |
| `settling_time` | 最后一次 `abs(sp - pv) > settle_band` 后的时间 |
| `final_abs_error` | 最后一个采样点的绝对误差 |

## 小例子

假设一条回路：

```text
pv = [0, 6, 11, 10]
sp = [10, 10, 10, 10]
lsl = 8, usl = 12
sample_interval = 1
settle_band = 1
```

误差为：

```text
error = sp - pv = [10, 4, -1, 0]
abs_error = [10, 4, 1, 0]
```

因此：

```text
IAE = 10 + 4 + 1 + 0 = 15
ISE = 100 + 16 + 1 + 0 = 117
ITAE = 0*10 + 1*4 + 2*1 + 3*0 = 6
MAE = 15 / 4 = 3.75
RMSE = sqrt(117 / 4) = 5.408
final_abs_error = 0
```

越限点为 `0` 和 `6`：

```text
out_of_spec_count = 2
out_of_spec_ratio = 2 / 4 = 0.5
```

规格宽度为 `4`，所以：

```text
overshoot_ratio = max(pv - sp, 0) / 4 = 1 / 4 = 0.25
undershoot_ratio = max(sp - pv, 0) / 4 = 10 / 4 = 2.5
```

最后一次 `abs_error > 1` 出现在 `k=1`，所以 `settling_time = 2`。

## 构建

```bash
cd prediction/ProcessControl/PIDModelFit/pid_control_performance_metrics
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
make -j$(nproc)
```

## 测试

```bash
python tests/test_pid_control_performance_metrics.py
python tests/benchmark_pid_control_performance_metrics.py
```

NPU smoke：

```bash
./build/test_aclnn_pid_control_performance_metrics 3
```

NPU/CPU benchmark：

```bash
./build/benchmark_pid_control_performance_metrics 3 128 4096 20 3 64
```

## 文档

- [算法说明](docs/algorithm.md)
- [API 说明](docs/api_reference.md)
- [测试报告](docs/benchmark.md)
