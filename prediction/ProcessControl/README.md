# 工业过程控制算子（Process Control）

本目录面向化工、能源、材料制造中的工业过程控制场景，聚焦模型辨识、控制器整定、在线预测与优化调度等任务。

## 当前算子

| 方向 | 算子/模型 | 状态 | 说明 |
|------|-----------|------|------|
| PID 模型辨识 | [PIDModelFit](PIDModelFit/README.md) | ✅ Ascend C 就绪 | 面向 FOPDT/IPDT/SOPDT 低阶过程模型的多回路、多候选并行辨识（12 个算子） |
| 时序预测模型 | [TimeSeriesForecast](TimeSeriesForecast/README.md) | ✅ Ascend C 就绪 | 面向 SSM/Mamba、Autoformer/Reformer 等预测模型的高价值 fused operators（13 个算子） |
| 非时序预测模型 | [NonTemporalPrediction](NonTemporalPrediction/) | ✅ Ascend C 就绪 | 面向图学习（GAT/GATv2/GraphSAGE/LightGCN 等）、点云（PointNet++/PPFNet）与分子图几何（DimeNet/GemNet）的 CSR 稀疏传播与注意力聚合 fused operators（20 个算子，目录暂无汇总 README，见各算子子目录） |

> ✅ Ascend C 就绪 = 已完成 Ascend C 算子开发，含完整测试（状态含义与仓库根 README 一致）。

## 典型场景

- 批量回路整定：一次性对装置中多条温度、压力、流量、液位回路进行候选模型筛选。
- 在线自整定：滚动窗口数据到达后，快速刷新过程模型参数，为 PID 参数整定提供基础。
- 仿真平台评估：在工艺仿真、数字孪生或控制策略搜索中，对大量候选参数进行批量打分。
- 在线时序预测：针对 torch_npu/PyTorch 框架路径中无法高效表达的 scan、autocorrelation 和 LSH 分桶子图提供 fused operator。
- 非时序图学习与几何推理：为图神经网络的稀疏传播/注意力聚合、点云邻域查询与分子图多体几何计算提供可复用的 NPU fused operator。
