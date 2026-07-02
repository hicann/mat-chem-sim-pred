# PID 整定算子组

本目录提供面向工业过程控制 PID 整定链路的 Ascend C 算子组。当前正式提交 **12 个独立算子**，覆盖从过程模型辨识、模型质量诊断、PID 参数生成、候选闭环仿真与评分，到控制性能和过程能力评估的主要环节。

整条工程链路可以概括为：

```text
采集数据 -> 模型辨识 -> 残差诊断 -> PID 参数生成 -> 候选仿真/评分/选优 -> 性能/能力评估
```

每个 `pid_*` 目录都是独立算子工程，拥有独立的 `CMakeLists.txt`、`op_host/`、`op_kernel/`、`tests/`、`examples/` 和 README。顶层目录不提供统一 CMake，公共 Python reference 和共享头文件放在 `common/`，全链路验证工具放在 `e2e/`。

## 算子总览

| 阶段 | 算子目录 | Host API | 作用 |
|------|----------|----------|------|
| 模型辨识 | [pid_fopdt_basis_gemm_fit](pid_fopdt_basis_gemm_fit/README.md) | `aclnnPidFopdtBasisGemmFit` | FOPDT 一阶惯性加纯滞后模型 basis-GEMM 辨识 |
| 模型辨识 | [pid_ipdt_basis_gemm_fit](pid_ipdt_basis_gemm_fit/README.md) | `aclnnPidIpdtBasisGemmFit` | IPDT 积分加纯滞后模型 basis-GEMM 辨识 |
| 模型辨识 | [pid_sopdt_basis_gemm_fit](pid_sopdt_basis_gemm_fit/README.md) | `aclnnPidSopdtBasisGemmFit` | SOPDT 二阶惯性加纯滞后模型 basis-GEMM 辨识 |
| 模型诊断 | [pid_residual_diagnostics](pid_residual_diagnostics/README.md) | `aclnnPidResidualDiagnostics` | 全序列残差均值、RMSE、Durbin-Watson、多 lag 自相关和白噪声诊断 |
| 模型诊断 | [pid_windowed_residual_diagnostics](pid_windowed_residual_diagnostics/README.md) | `aclnnPidWindowedResidualDiagnostics` | 滑窗残差诊断，用于暴露分段漂移和局部模型失配 |
| PID 参数生成 | [pid_tuning_rule_batch](pid_tuning_rule_batch/README.md) | `aclnnPidTuningRuleBatch` | 对 FOPDT 参数批量生成 Ziegler-Nichols、IMC、Cohen-Coon 三类 PID 参数 |
| 候选仿真 | [pid_fopdt_batch_rollout_score](pid_fopdt_batch_rollout_score/README.md) | `aclnnPidFopdtBatchRolloutScore` | FOPDT 模型的批量闭环 rollout 评分与最优 PID 选择 |
| 候选仿真 | [pid_ipdt_batch_rollout_score](pid_ipdt_batch_rollout_score/README.md) | `aclnnPidIpdtBatchRolloutScore` | IPDT 模型的批量闭环 rollout 评分与最优 PID 选择 |
| 候选仿真 | [pid_sopdt_batch_rollout_score](pid_sopdt_batch_rollout_score/README.md) | `aclnnPidSopdtBatchRolloutScore` | SOPDT 模型的批量闭环 rollout 评分与最优 PID 选择 |
| 候选特征 | [pid_step_response_features](pid_step_response_features/README.md) | `aclnnPidStepResponseFeatures` | 从 PID 候选阶跃响应轨迹提取 rise、settling、overshoot、IAE、ISE 等特征 |
| 性能评估 | [pid_control_performance_metrics](pid_control_performance_metrics/README.md) | `aclnnPidControlPerformanceMetrics` | 批量计算 Harris、IAE、ISE、超调、稳定时间等控制性能指标 |
| 能力评估 | [pid_process_capability_metrics](pid_process_capability_metrics/README.md) | `aclnnPidProcessCapabilityMetrics` | 批量计算均值、标准差、Cp、Cpk、Pp、Ppk 和越限比例 |

## 核心计算路线

模型辨识阶段采用 basis-GEMM 路线。FOPDT、IPDT、SOPDT 三类模型先把候选参数网格转换为单位增益响应基函数矩阵 `basis_t[N, M]`，再对 `B` 条回路的中心化输出 `y_centered[B, N]` 执行：

```text
dot[B, M] = y_centered[B, N] x basis_t[N, M]
K_hat[b, m] = dot[b, m] / basis_norm[m]
SSE[b, m] = y_energy[b] - dot[b, m]^2 / basis_norm[m]
best[b] = argmin_m SSE[b, m]
```

其中 `dot` 由 CANN 内置 MatMul 完成，自定义算子负责增益估计、SSE 评分和 best-candidate reduce，输出 `best_sse[B]`、`best_k[B]` 和 `best_idx[B]`。这一路线的价值在于多回路共享基函数后可以塌缩为大 GEMM，`dot[B,M]` 保持在 device 侧直接接后续 reduce。

下游整定阶段分为两类：

- 规则整定：`pid_tuning_rule_batch` 根据模型参数直接生成经验 PID 参数，是当前 FOPDT E2E 链路中的候选生成阶段；它不作为性能主线宣传，但保留为端到端完整性的一环。
- 仿真选优：`pid_*_batch_rollout_score` 在候选 PID 维度做批量闭环递推，融合候选特征、评分和 best 选择，输出每条回路的最优候选。`pid_step_response_features` 保留为候选轨迹可解释特征提取算子。

诊断和评估算子适合接在 device-resident 流水线中作为后处理，减少中间轨迹或中间统计量回传 Host 的成本。

## 构建与测试

每个算子独立构建。以 `pid_fopdt_basis_gemm_fit` 为例：

```bash
cd prediction/ProcessControl/PIDModelFit/pid_fopdt_basis_gemm_fit
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j$(nproc)
```

常用测试入口：

```bash
# Python reference 精度测试
python tests/test_<op>.py

# ACL/NPU smoke，具体可执行文件名见各算子 README
./build/test_aclnn_<op> 0

# NPU/CPU benchmark，参数见各算子 README
./build/benchmark_<op> ...
```

全链路验证工具位于 [e2e](e2e/README.md)，用于串联真实算子并和 CPU reference 对齐。公共 NumPy reference 位于 [common](common/)。

## 验证结论与工程定位

已在 node202 / Ascend910B3 环境上完成多批构建、smoke、Python reference 和 benchmark 验证。当前较稳定的结论如下：

- 模型辨识是最明确的 NPU 价值点。`B=64,N=1024,M=256` 下，FOPDT/IPDT/SOPDT 的 `MatMul + reduce` resident e2e 约 `0.17-0.30 ms`，cold e2e 约 `0.39-0.99 ms`，相对 CPU 64 线程完整 fit 有明显优势，且 `idx_diff=0`。
- 全链路 device-resident 方案可以把辨识、候选仿真和后处理串起来，中间张量留在 device 侧。实测显示 rollout 阶段通常占 E2E 主要耗时，因此全链路加速会随候选数和仿真步数变化。
- 残差诊断、控制性能、过程能力、候选特征等归约/扫描类算子更适合挂在 NPU resident 链路后端；单独从 Host 调用时，H2D/D2H 可能压平 kernel 优势。
- `pid_tuning_rule_batch` 算术强度较低，不适合作为单独性能卖点，但它支撑 E2E 中“模型参数 -> PID 候选”的规则整定阶段，因此保留在正式提交中。
- `pid_basis_gemm_reduce_tiled`、`pid_fopdt_closed_loop_score`、`pid_frequency_stability_scan`、`pid_control_candidate_score`、`pid_control_candidate_best` 五个探索/历史原型不进入本次正式 12 算子提交；相关方向可在后续完成单 kernel 化、向量化或语义收敛后再评估。

更详细的设计、测试和性能材料见：

- [设计文档](docs/design.md)
- [测试报告](docs/test_report.md)
- [算子用途分类](docs/pid_operator_catalog_by_purpose.md)

各算子的数学定义、API、benchmark 细节位于对应算子目录下的 `docs/algorithm.md`、`docs/api_reference.md` 和 `docs/benchmark.md`。
