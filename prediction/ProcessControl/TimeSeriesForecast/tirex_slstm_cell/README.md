# TirexSlstmCell 迁移交付目录

本目录从 C:/tslib/deliverables/tirex_slstm_cell_vectorized_experimental 迁移，按 PIDModelFit/CANN_OPERATOR_RULES.md 整理为本仓库时序预测算子交付结构。

## 状态

| 项目 | 状态 |
|------|------|
| msopgen | msopgen/tirex_slstm_cell_msopgen.json |
| Ascend C kernel | 已迁入 op_kernel/ |
| op_host | 已迁入 op_host/ |
| tests | 已迁入 tests/ |
| source docs | 已保留到 docs/source_* |

## 说明

完整算法、API、benchmark 和测试状态分别见 docs/algorithm.md、docs/api_reference.md、docs/benchmark.md、docs/test_report.md。

## TorchAir 补充结论

当前软件栈在 sLSTM 原表达的 `aten.log_sigmoid_forward.default` 处缺少 converter，主测试 `B64,S64,H512,heads4` 无法直接完整成图。将 logsigmoid 人工等价改写为稳定的基础张量操作后，TorchAir 为 `6.318 ms`，custom 为 `3.542 ms`，custom 仍快 `1.78x`；首次编译和运行耗时 `158.61 s`，max diff `7.45e-08`。custom 直接实现 sLSTM cell 语义，接入时不需要为了 converter 缺失重写 logsigmoid 表达。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
