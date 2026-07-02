# PIDModelFit 12 算子测试报告

## 测试对象

当前正式提交 12 个算子，覆盖模型辨识、模型诊断、PID 参数生成、候选仿真、候选特征、控制性能和过程能力评估。

| 阶段 | 算子 | Host API |
|------|------|----------|
| 模型辨识 | `pid_fopdt_basis_gemm_fit` | `aclnnPidFopdtBasisGemmFit` |
| 模型辨识 | `pid_ipdt_basis_gemm_fit` | `aclnnPidIpdtBasisGemmFit` |
| 模型辨识 | `pid_sopdt_basis_gemm_fit` | `aclnnPidSopdtBasisGemmFit` |
| 模型诊断 | `pid_residual_diagnostics` | `aclnnPidResidualDiagnostics` |
| 模型诊断 | `pid_windowed_residual_diagnostics` | `aclnnPidWindowedResidualDiagnostics` |
| PID 参数生成 | `pid_tuning_rule_batch` | `aclnnPidTuningRuleBatch` |
| 候选仿真 | `pid_fopdt_batch_rollout_score` | `aclnnPidFopdtBatchRolloutScore` |
| 候选仿真 | `pid_ipdt_batch_rollout_score` | `aclnnPidIpdtBatchRolloutScore` |
| 候选仿真 | `pid_sopdt_batch_rollout_score` | `aclnnPidSopdtBatchRolloutScore` |
| 候选特征 | `pid_step_response_features` | `aclnnPidStepResponseFeatures` |
| 控制性能 | `pid_control_performance_metrics` | `aclnnPidControlPerformanceMetrics` |
| 过程能力 | `pid_process_capability_metrics` | `aclnnPidProcessCapabilityMetrics` |

## 测试环境

| 项目 | 环境 |
|------|------|
| 本地开发机 | Windows / Python 3.11 |
| NPU 测试机 | node202 |
| NPU | Ascend 910B3 |
| CANN | node202 `/usr/local/Ascend/ascend-toolkit` |
| 构建工具 | CMake + Ascend C CMake |
| CPU 对照 | benchmark 程序内置 C++/Python reference，典型口径为 64 线程 |

## 文档与工程完整性

12 个正式算子均提供独立工程和基础交付件：

| 检查项 | 结果 |
|--------|------|
| 独立 `CMakeLists.txt` | 12/12 |
| `op_host/` + `op_kernel/` | 12/12 |
| README | 12/12 |
| `docs/algorithm.md` | 12/12 |
| `docs/api_reference.md` | 12/12 |
| `docs/benchmark.md` | 12/12 |
| Python reference / accuracy test | 12/12 |
| ACLNN smoke 或 NPU benchmark 入口 | 12/12 |

## 构建与基础验证

单算子构建方式：

```bash
cd prediction/ProcessControl/PIDModelFit/<op>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j$(nproc)
```

常用验证入口：

```bash
# Python reference 精度测试
python tests/test_<op>.py

# NPU smoke，具体可执行文件名见各算子 README
./build/test_aclnn_<op> 0

# NPU/CPU benchmark，参数见各算子 docs/benchmark.md
./build/benchmark_<op> ...
```

Python 语法检查：

```bash
python -m compileall -q prediction/ProcessControl/PIDModelFit
```

## 精度验证结论

| 算子类别 | 精度验证结论 |
|----------|--------------|
| FOPDT/IPDT/SOPDT basis-GEMM fit | pipeline benchmark 中 `idx_diff=0`，`best_sse/best_k` 与 CPU reference 在 float32 阈值内对齐 |
| 残差诊断 | smoke 通过；benchmark 中 `max_autocorr_abs_err=0`，指标误差在 float32 归约阈值内 |
| 窗口残差诊断 | Python reference `7 passed`；NPU benchmark 中 `metric_max_abs <= 7.63e-6`，`autocorr_max_abs=0` |
| PID 规则整定 | 大 batch 精度 `pid_max_abs=0`、`diag_max_abs=0`；用于 E2E 候选生成 |
| FOPDT rollout | NPU 输出与 CPU reference bit-identical；大候选下个别 `best_idx` 差异来自 argmin 近并列 |
| IPDT/SOPDT rollout | 质量指标与 CPU reference 在 float32 精度内；少量 `best_idx` 差异对应近并列候选，得分相对误差小 |
| 阶跃响应特征 | 本地 reference `4 passed`；NPU benchmark `feature_max_abs=0` |
| 控制性能指标 | 与 CPU double reference 最大相对误差保持在 `1e-4` 以内 |
| 过程能力指标 | 与 CPU Welford reference 对齐，误差符合 float32 累加和近似开方预期 |

## 性能验证结论

性能细节以各算子目录的 `docs/benchmark.md` 为准。当前可用于合入判断的代表性结论如下。

### 模型辨识

`MatMul + custom reduce` pipeline 在 `B=64,N=1024,M=256` 下对 CPU 64 线程完整 fit 的加速：

| 算子 | CPU 64T 完整 fit | NPU resident e2e | NPU cold e2e | resident 加速 | cold 加速 |
|------|------------------|------------------|--------------|---------------|-----------|
| FOPDT fit | `8.74037 ms` | `0.303587 ms` | `0.989354 ms` | `28.79x` | `8.83x` |
| IPDT fit | `8.85539 ms` | `0.185910 ms` | `0.389072 ms` | `47.63x` | `22.76x` |
| SOPDT fit | `10.3824 ms` | `0.166816 ms` | `0.406088 ms` | `62.24x` | `25.57x` |

FOPDT 扩展规模 `B=128,N=1024,M=512` 下 resident/cold 加速分别为 `92.87x` / `18.73x`，说明候选数和回路数扩大后矩阵化路线收益更明显。

### 候选闭环仿真

`B=128,sim_steps=1024` 口径下，三个 batch rollout 算子相对 CPU 64 线程的 kernel 加速：

| 算子 | `C=1024` | `C=4096` | `C=16384` |
|------|----------|----------|-----------|
| FOPDT rollout | 约 `4.4x` | 约 `5.3x-6.8x` | 约 `5.1x` |
| IPDT rollout | `4.36x` | `4.95x` | `4.55x` |
| SOPDT rollout | `5.04x` | `5.07x` | `5.18x` |

rollout 是整定链路中的主要热点。当前实现利用候选维度宽 SIMD 并行，在保持模型递推语义不变的前提下获得稳定单卡收益。

### 诊断、特征与评估

| 算子 | 代表性结论 |
|------|------------|
| `pid_residual_diagnostics` | 中大规模下 e2e 可超过 CPU 64 线程，`B=1024,N=2048,max_lag=64` e2e/CPU64T 为 `4.66x` |
| `pid_windowed_residual_diagnostics` | resident e2e 加速约 `31x-64x`，适合上游预测值已在 device 的窗口化质量门禁 |
| `pid_step_response_features` | resident e2e 加速约 `41x-43x`，特征与 CPU reference 完全对齐 |
| `pid_control_performance_metrics` | kernel-only 相对 CPU 64T 约 `16x-100x`，适合作为 NPU 流水线融合后处理 |
| `pid_process_capability_metrics` | kernel-only 相对 CPU fast 64T 约 `2.1x-11.5x`，适合 device-resident 巡检场景 |
| `pid_tuning_rule_batch` | 低算术强度算子，不作为性能主卖点；保留原因是 E2E 需要 device 侧候选生成阶段 |

## E2E 验证

全链路工具位于 `e2e/`，覆盖 FOPDT 方向的真实算子串联：

```text
fit -> tuning_rule -> fopdt_rollout -> performance_metrics
```

`e2e_orchestrator.py` 用 CPU reference 分阶段校验精度；`e2e_perf` 用单进程 device-resident 编排计时，并与 CPU 64 线程链路对比。代表性结果见 `e2e/README.md`：`B=128,sim_steps=1024,auto tile` 下，`C=1024/4096/16384` 分别约 `4.0x/6.2x/4.5x` vs CPU 64T。

## 结论

- 12 个算子均具备独立工程、文档、reference 测试和 NPU benchmark/smoke 入口。
- 模型辨识与 batch rollout 是最主要的 NPU 价值点：前者利用 GEMM，后者利用候选维度宽 SIMD。
- 残差诊断、窗口残差、候选特征、控制性能和过程能力算子适合作为 device-resident 后处理，避免中间轨迹回传 CPU。
- `pid_tuning_rule_batch` 不是性能卖点，但支撑端到端流程完整性。
- 已撤回的探索/历史原型不在本次正式提交范围内，当前目录只保留 12 个正式算子。
