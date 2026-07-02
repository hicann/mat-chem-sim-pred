# PidStepResponseFeatures Delivery Note

## 结论

`PidStepResponseFeatures` 是本轮更合适的“新算子”交付目标。它此前只有 Python reference、测试和本机 benchmark，本轮新增了 Ascend C host/kernel/CMake/smoke/benchmark，使其成为可构建、可运行、可验证的 NPU 原型算子。

## 这个算子是干啥的

它用于 PID 候选控制轨迹的批量后处理。上游 rollout、仿真或预测已经生成：

```text
pv_candidates[B, C, N], sp[B, N]
```

该算子在 NPU 上把每条候选轨迹压缩为：

```text
features[B, C, 12]
```

输出特征包括初值、终值、终值误差、峰值、谷值、超调、欠调、上升时间、峰值时间、调节时间、IAE 和 ISE。下游可以直接用这些特征做候选筛选、加权评分或 best candidate reduce。

## 为什么值得做

- 它不做闭环递推，不受强时序依赖限制，适合把上游轨迹当作 device-resident 数据做后处理。
- 工作量随 `B * C * N` 增长，输出只有 `B * C * 12`，适合 NPU 侧融合扫描。
- 它可以替代“把完整候选轨迹搬回 CPU 再逐候选算特征”的流程，减少 D2H 数据量。
- 它和后续候选评分方向互补：先提取可解释特征，再由 host 策略或未来 fused scorer 做权重评分和 best reduce。

## 本次交付

- `op_host/pid_step_response_features_host.h`
- `op_host/pid_step_response_features_host.cpp`
- `op_host/pid_step_response_features_def.cpp`
- `op_kernel/pid_step_response_features_kernel.cpp`
- `examples/test_aclnn_pid_step_response_features.cpp`
- `tests/benchmark_pid_step_response_features_aclnn.cpp`
- `CMakeLists.txt`
- `README.md`
- `docs/algorithm.md`
- `docs/api_reference.md`
- `docs/benchmark.md`

## 验收结果

本地 Python reference：

```text
4 passed
```

node202 / Ascend910B3：

```text
CANN build: PASSED
ACLNN smoke: PASSED
```

Smoke 输出：

```text
PidStepResponseFeatures smoke initial=0 final=10 peak=11 overshoot=0.1 rise_time=2 peak_time=4 settling_time=5 iae=15 ise=91
PASSED
```

Benchmark：

```text
B=64 C=32 N=1024
cpu_64T_ms=4.41643
npu_kernel_ms=0.0547166
npu_resident_e2e_ms=0.101893
feature_max_abs=0

B=128 C=64 N=1024
cpu_64T_ms=6.87986
npu_kernel_ms=0.0792977
npu_resident_e2e_ms=0.164775
feature_max_abs=0
```

## 边界说明

- 当前 benchmark 是 resident 口径，假设 `pv_candidates/sp` 已在 Device。
- cold H2D 端到端尚未统计，完整候选轨迹来自 Host 时需要单独评估搬运成本。
- 当前 kernel 是 GlobalTensor 标量扫描原型，后续可继续做 UB 分块、向量化和与候选评分的融合。
