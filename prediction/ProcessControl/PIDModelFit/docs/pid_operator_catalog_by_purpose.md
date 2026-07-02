# PID 算子用途分类

整条 PID 工程链路可以按业务语义分为：**辨识模型 -> 检验模型 -> 生成 PID 候选 -> 仿真选优 -> 评估控制效果和过程能力**。

本目录正式提交 12 个 `pid_*` 算子，按用途归类如下。

## 1. 模型辨识

从 `pv/mv` 阶跃或运行数据拟合低阶过程模型，输出每条回路的最优候选模型下标、增益和 SSE。

| 算子 | 作用 |
|------|------|
| `pid_fopdt_basis_gemm_fit` | 一阶惯性加纯滞后模型 FOPDT 的 basis-GEMM 辨识 |
| `pid_ipdt_basis_gemm_fit` | 积分加纯滞后模型 IPDT 的 basis-GEMM 辨识 |
| `pid_sopdt_basis_gemm_fit` | 二阶惯性加纯滞后模型 SOPDT 的 basis-GEMM 辨识 |

这三个算子共享同一套最小二乘 reduce 公式，但按模型族保持独立入口、独立工程和独立文档。

## 2. 模型质量诊断

辨识完成后，用实测曲线与模型预测曲线的残差判断模型是否可信。

| 算子 | 作用 |
|------|------|
| `pid_residual_diagnostics` | 全序列残差均值、MAE、RMSE、最大残差、fit percent、Durbin-Watson、Ljung-Box 和多 lag 自相关 |
| `pid_windowed_residual_diagnostics` | 按滑动窗口输出同类残差指标，用于发现局部漂移、局部震荡和分段模型失配 |

残差诊断不是重复计算 SSE。SSE 衡量候选模型拟合误差总量，残差诊断进一步检查误差是否有偏、是否自相关、是否集中在某个局部窗口；这些信息可用于拒绝低质量辨识结果、切换模型族或要求重新采样。

## 3. PID 参数生成

从已辨识的模型参数生成初始 PID 候选。

| 算子 | 作用 |
|------|------|
| `pid_tuning_rule_batch` | 输入 FOPDT 模型参数，批量输出 Ziegler-Nichols、IMC、Cohen-Coon 三套 `Kp/Ki/Kd` 候选和诊断信息 |

该算子算术强度低，不作为性能主卖点；它的价值在于支撑端到端流程里“模型参数 -> PID 候选”的 device 侧闭环。

## 4. 候选仿真、评分与选优

对多组候选 PID 参数做闭环仿真，比较响应质量并选择最优候选。

| 算子 | 作用 |
|------|------|
| `pid_fopdt_batch_rollout_score` | 在 FOPDT 模型上对候选 PID 做批量 rollout、评分和 best 选择 |
| `pid_ipdt_batch_rollout_score` | 在 IPDT 模型上对候选 PID 做批量 rollout、评分和 best 选择 |
| `pid_sopdt_batch_rollout_score` | 在 SOPDT 模型上对候选 PID 做批量 rollout、评分和 best 选择 |
| `pid_step_response_features` | 从候选阶跃响应轨迹中提取 overshoot、rise time、settling time、IAE、ISE 等可解释特征 |

三个 batch rollout 算子已经融合了候选特征、候选评分和候选选优；如果直接使用其 `best_result/best_idx` 输出，通常不需要再接独立候选评分或 best 选择算子。`pid_step_response_features` 面向需要保留完整候选轨迹特征表的模块化链路。

## 5. 控制性能与过程能力

在整定结果上线前或运行巡检中，对控制效果和过程稳定性做批量评价。

| 算子 | 作用 |
|------|------|
| `pid_control_performance_metrics` | 基于 `pv/sp/mv` 计算 Harris、IAE、ISE、ITAE、MAE、RMSE、超调、稳定时间等控制性能指标 |
| `pid_process_capability_metrics` | 基于 `pv/lsl/usl` 计算均值、标准差、Cp、Cpk、Pp、Ppk、越限比例、min/max 等过程能力指标 |

控制性能关注“跟踪设定值好不好”，过程能力关注“是否稳定落在规格范围内”。二者可以同时用于上线验收：前者评价控制器行为，后者评价过程质量。
