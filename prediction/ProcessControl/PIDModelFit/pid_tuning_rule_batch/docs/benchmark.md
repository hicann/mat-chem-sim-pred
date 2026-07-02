# PidTuningRuleBatch Benchmark Report

## 测试环境

- NPU 环境：node202
- SOC：Ascend910B3
- NPU device：3
- CANN：`/usr/local/Ascend/ascend-toolkit`
- CPU 对比线程数：64
- 数据类型：float32

## 测试口径

- `cpu_single_ms`：单线程 C++ reference。
- `cpu_64_ms`：64 线程 C++ reference。
- `npu_kernel_ms`：输入已在 Device，仅统计算子执行时间。
- `npu_e2e_ms`：包含 4 个输入 H2D、kernel、2 个输出 D2H。

输出：

```text
pid_params[B, 3, 3]      # Kp, Ki, Kd
diagnostics[B, 3, 4]     # valid, dead_time_ratio, aggressiveness, lambda_ratio
```

## 正确性

Smoke：

```text
PidTuningRuleBatch smoke passed.
```

Python reference 覆盖：

- vectorized reference 与逐条 loop reference 对齐。
- 固定手算工况校验 Ziegler-Nichols、IMC、Cohen-Coon 输出。
- 无效输入被过滤，`pid_params` 清零且 `valid=0`。
- 输出 shape 与有限性检查通过。

NPU 版本采用单核 launch 写回 `pid_params/diagnostics`，避免轻量标量 `SetValue` 多核写同一 cache line 时可能出现的写回一致性问题；这类规则整定计算本身很轻，单核写回对 E2E 主链影响很小。

## 性能数据

| B | CPU single | CPU 64T | NPU kernel | NPU e2e | e2e/CPU64 |
|---:|---:|---:|---:|---:|---:|
| 4096 | 0.275839 ms | 0.455669 ms | 0.0947986 ms | 0.170780 ms | 2.67x |
| 8192 | 0.561186 ms | 0.525737 ms | 0.188849 ms | 0.261154 ms | 2.01x |
| 16384 | 1.11962 ms | 0.905922 ms | 0.179909 ms | 0.248650 ms | 3.64x |
| 65536 | 5.10277 ms | 0.503638 ms | 0.602354 ms | 1.38543 ms | 0.36x |
| 262144 | 24.7467 ms | 0.724268 ms | 2.35919 ms | 4.96856 ms | 0.15x |
| 1048576 | 85.5412 ms | 3.31345 ms | 9.32408 ms | 26.1733 ms | 0.13x |

## 结论

`PidTuningRuleBatch` 是低算术强度的闭式公式算子。小 batch 下 NPU e2e 可以快于 CPU 64T；大 batch 下 CPU 多线程对这类简单公式已经很高效，单独调用时 NPU e2e 并不总是占优。

因此该算子的推荐定位是 FOPDT E2E 链路中的 device-side 候选生成阶段，而不是独立性能卖点。真正决定整定链路性能的阶段仍是 basis-GEMM 模型辨识、批量闭环 rollout 和后续融合指标计算。
