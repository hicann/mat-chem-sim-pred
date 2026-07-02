# PID 整定算子组设计文档

## 目标

`PIDModelFit` 面向工业过程控制中的批量 PID 整定链路。当前正式提交 12 个 Ascend C 算子，目标不是把某一个小公式搬到 NPU 上，而是让“模型辨识、模型诊断、PID 候选生成、候选闭环仿真、性能/能力评估”可以在 device 侧串成流水线，减少中间轨迹和中间统计量在 Host 与 Device 之间反复搬移。

整体流程如下：

```text
pv/sv/mv 采集数据
  -> FOPDT/IPDT/SOPDT 模型辨识
  -> 残差诊断与窗口化残差诊断
  -> PID 规则整定生成候选
  -> 候选闭环 rollout 仿真、评分、选优
  -> 控制性能与过程能力评估
```

## 算子分组

| 阶段 | 算子 | 设计定位 |
|------|------|----------|
| 模型辨识 | `pid_fopdt_basis_gemm_fit` | FOPDT 候选模型的 basis-GEMM 最小二乘辨识 |
| 模型辨识 | `pid_ipdt_basis_gemm_fit` | IPDT 候选模型的 basis-GEMM 最小二乘辨识 |
| 模型辨识 | `pid_sopdt_basis_gemm_fit` | SOPDT 候选模型的 basis-GEMM 最小二乘辨识 |
| 模型诊断 | `pid_residual_diagnostics` | 全序列残差指标、自相关和白噪声诊断 |
| 模型诊断 | `pid_windowed_residual_diagnostics` | 滑窗残差诊断，发现局部漂移、局部振荡和分段失配 |
| PID 参数生成 | `pid_tuning_rule_batch` | 根据模型参数批量生成 Ziegler-Nichols、IMC、Cohen-Coon 三类候选 PID |
| 候选仿真 | `pid_fopdt_batch_rollout_score` | 在 FOPDT 模型上批量仿真候选 PID，融合评分和 best 选择 |
| 候选仿真 | `pid_ipdt_batch_rollout_score` | 在 IPDT 模型上批量仿真候选 PID，融合评分和 best 选择 |
| 候选仿真 | `pid_sopdt_batch_rollout_score` | 在 SOPDT 模型上批量仿真候选 PID，融合评分和 best 选择 |
| 候选特征 | `pid_step_response_features` | 将候选阶跃响应轨迹压缩为 rise/settling/overshoot/IAE/ISE 等特征 |
| 控制性能 | `pid_control_performance_metrics` | 对 `pv/sp/mv` 批量计算 Harris、IAE、ISE、超调、稳定时间等 20 个指标 |
| 过程能力 | `pid_process_capability_metrics` | 对运行窗口批量计算均值、标准差、Cp、Cpk、Pp、Ppk 和越限比例 |

每个 `pid_*` 目录都是独立算子工程，包含独立 `CMakeLists.txt`、`op_host/`、`op_kernel/`、`tests/`、`examples/`、README 和算子内文档。公共 Python reference、共享校验函数和复用头文件放在 `common/`。

## 模型辨识设计

FOPDT、IPDT、SOPDT 三个模型辨识算子采用相同的矩阵化路线。对每个候选模型参数，先根据 `mv` 输入轨迹预生成单位增益响应基函数 `basis[n,m]`；对每条回路的实测输出做中心化得到 `y_centered[b,n]`。核心计算为：

```text
dot[B, M] = y_centered[B, N] x basis_t[N, M]
basis_norm[m] = sum_n basis[n,m]^2
y_energy[b] = sum_n y_centered[b,n]^2
K_hat[b,m] = dot[b,m] / basis_norm[m]
SSE[b,m] = y_energy[b] - dot[b,m]^2 / basis_norm[m]
best_idx[b] = argmin_m SSE[b,m]
```

`dot[B,M]` 由 CANN 内置 MatMul 完成，自定义算子负责 `K_hat/SSE/best_idx` 的融合 reduce。这样既能利用 NPU cube 做大矩阵乘，也避免把完整 `SSE[B,M]` 写回 Host；后续只需要每条回路的 `best_sse`、`best_k` 和 `best_idx`。

## 下游整定设计

模型辨识输出的是过程模型参数，不是 PID 参数本身。下游分三步：

1. `pid_tuning_rule_batch` 根据 FOPDT 参数批量生成规则整定候选，输出 `pid_params[B,3,3]`，第二维对应 Ziegler-Nichols、IMC、Cohen-Coon。
2. `pid_*_batch_rollout_score` 接收过程模型参数和一组候选 `kp/ki/kd`，在候选维度做批量闭环递推，融合候选特征、质量评分和最优候选选择，输出每条回路的 `best_result[B,8]` 和 `best_idx[B]`。
3. `pid_step_response_features`、`pid_control_performance_metrics`、`pid_process_capability_metrics` 用于需要保留轨迹解释、上线验收或运行巡检的场景。

三类 rollout 算子的区别来自过程模型递推方程：FOPDT 是一阶惯性加滞后，IPDT 是积分加滞后，SOPDT 是二阶惯性加滞后；PID 控制律、候选评分和 best 选择语义保持一致。

## 诊断与评估设计

残差诊断回答“辨识出的模型是否足够可信”。如果残差均值偏离 0，说明模型存在系统性偏差；如果残差自相关和 Ljung-Box 统计量偏大，说明残差不是白噪声，模型仍未解释掉某些动态结构。窗口化版本进一步按时间窗口检查局部失配，适合发现漂移、局部振荡或工况切换。

控制性能指标回答“控制器跟踪设定值的效果如何”，关注 `pv-sp` 误差、超调、稳定时间、IAE/ISE 等。过程能力指标回答“控制后的过程是否满足规格上下限”，关注 `Cp/Cpk/Pp/Ppk`、越限比例和均值偏移。两者关注点不同，因此作为两个独立算子保留。

## NPU 工程定位

- 模型辨识是明确的 NPU 主线：多回路、多候选天然形成 `B x N` 与 `N x M` 的 GEMM，resident 与 cold 口径下均显著快于 CPU 64 线程完整 fit。
- rollout 是候选整定阶段的主要热点：虽然时间递推在采样步上串行，但候选维度可用宽 SIMD 并行，FOPDT/IPDT/SOPDT 三个 rollout 算子在典型候选规模下相对 CPU 64 线程有约 4x 到 7x 的 kernel 加速。
- 残差、窗口残差、候选特征和指标类算子适合作为 device-resident 后处理；当上游数据已经在 NPU 上时，可避免把整条轨迹搬回 CPU。
- `pid_tuning_rule_batch` 算术强度较低，不作为单独性能卖点；它保留的原因是端到端链路需要一个 device 侧“模型参数 -> PID 候选”的规则整定阶段。

详细性能数据以各算子目录下的 `docs/benchmark.md` 为准。
