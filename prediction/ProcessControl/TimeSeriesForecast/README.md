# TimeSeriesForecast 时序预测 CANN 算子

本目录承接时序预测模型中框架路径难以高效表达的 fused operator。第一批提交只包含交付件相对完整、性能证据清楚、适合单独评审的 10 个算子。

## 第一批算子

| 算子目录 | 服务模型/模型族 | 主要价值 | 性能摘要 |
|---|---|---|---|
| [selective_scan_1d](selective_scan_1d/README.md) | Mamba / SSM | 将 torch_npu 中 Python loop + 多小算子的 selective scan 融合为单算子 | Mamba block E2E `9.02x`；真实 validation loop `10.22x` |
| [s6scan_fused](s6scan_fused/README.md) | S6 / Mamba-style SSM | 融合 selective SSM recurrence | component `136.56x`；E2E `44.15x` |
| [autocorr_fused_aggregate](autocorr_fused_aggregate/README.md) | Autoformer | 融合 lag scoring/top-k/softmax/aggregation | validation loop `10.06x`；aggregation `28.06x` |
| [batch_spd_inv_fp32](batch_spd_inv_fp32/README.md) | Koopa / DMD | 将 `torch.linalg.lstsq` CPU fallback 改为 on-device Gram inverse | DMD `12823.56x`；Koopa E2E `602.19x` |
| [tirex_slstm_cell](tirex_slstm_cell/README.md) | TiRex / xLSTM | 融合 sLSTM recurrent cell | layer/E2E `10.46x`；cell `14.31x-16.7x` |
| [cfc_scan_fused](cfc_scan_fused/README.md) | CfC / continuous-time RNN | 融合无法并行化的 per-timestep recurrence | component `84.83x`；E2E `70.26x` |
| [cornn_scan_fused](cornn_scan_fused/README.md) | coRNN | 融合二阶 oscillator recurrence | component `56.73x`；E2E `54.15x` |
| [sru_scan_fused](sru_scan_fused/README.md) | SRU / recurrent scan | 融合 input-only gate + peephole recurrence | component `163.48x`；E2E `71.46x` |
| [unicornn_scan_fused](unicornn_scan_fused/README.md) | UnICORNN | 融合 diagonal oscillatory recurrence | component `160.68x`；E2E `100.43x` |
| [ltc_scan_fused](ltc_scan_fused/README.md) | LTC / continuous-time RNN | 融合 `K * L` ODE unfold recurrence | component `82.44x`；E2E `76.90x` |

## 性能口径

本批次主要对比的是“不开发自定义算子时，使用 torch_npu/PyTorch 框架在 NPU 上表达同一子图”的路径，不把 CPU baseline 当作主要收益来源。

例外是 `BatchSpdInvFp32`：Koopa 原始 `torch.linalg.lstsq` 不支持 NPU，只能回退 CPU；TorchAir 也无法把该路径编译为 NPU 完整图。该算子的价值是补齐设备能力，使 DMD 求解能够在 NPU 上运行。DMD `B32,m7,E128` 相对 CPU fallback 提升 `12823.56x`，Koopa `seq336,pred96,B32` 端到端提升 `602.19x`。

## 交付结构

每个算子目录按 `prediction/ProcessControl/PIDModelFit/CANN_OPERATOR_RULES.md` 整理，包含：

```text
<op>/
├── CMakeLists.txt
├── README.md
├── docs/
│   ├── algorithm.md
│   ├── api_reference.md
│   ├── benchmark.md
│   └── test_report.md
├── op_host/
├── op_kernel/
├── examples/
└── tests/
```

第一批 10 个算子已在 Ascend 910B3 环境完成 msopgen/C++ 构建、ACLNN runtime、正确性和性能验证。TorchAir 与 custom 的统一结论见 [第一批 TorchAir 与自定义算子报告](docs/torchair_first_batch_report.md)。

使用大模型协作开发后续 CANN 算子时，可复用 [CANN 算子大模型协作开发提示词模板](CANN_OPERATOR_LLM_PROMPT_TEMPLATE.md)。模板覆盖需求分析、契约冻结、Host/tiling、Ascend C 实现、编译闭环、NPU 数值验证、性能优化、模型集成和提交审查。
