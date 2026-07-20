# 工业过程控制算子（Process Control）

本目录面向化工、能源、材料制造中的工业过程控制场景，聚焦模型辨识、控制器整定、在线预测与优化调度等任务。

## 当前算子

| 方向 | 算子/模型 | 状态 | 说明 |
|------|-----------|------|------|
| PID 模型辨识 | [PIDModelFit](PIDModelFit/README.md) | Ascend C 原型 | 面向 FOPDT/IPDT/SOPDT 低阶过程模型的多回路、多候选并行辨识 |
| 时序预测模型 | [TimeSeriesForecast](TimeSeriesForecast/README.md) | P0 已迁入 | 面向 SSM/Mamba、Autoformer/FEDformer 等预测模型的高价值 fused operators |

## 典型场景

- 批量回路整定：一次性对装置中多条温度、压力、流量、液位回路进行候选模型筛选。
- 在线自整定：滚动窗口数据到达后，快速刷新过程模型参数，为 PID 参数整定提供基础。
- 仿真平台评估：在工艺仿真、数字孪生或控制策略搜索中，对大量候选参数进行批量打分。
- 在线时序预测：针对 torch_npu/PyTorch 框架路径中无法高效表达的 scan、autocorrelation 和频域分解子图提供 fused operator。
