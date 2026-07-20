# TirexSlstmCell 性能报告

性能和 E2E 结果来自源交付物 C:/tslib/deliverables/tirex_slstm_cell_vectorized_experimental。本次迁移保留了源侧结果文件：

- docs/source_tirex_slstm_e2e_results.json
- docs/source_tirex_slstm_probe_results.json

迁移目录已在 Ascend 910B3 环境重新完成 msopgen 构建、ACLNN runtime 和正式 shape benchmark。

## 主要性能结果

主测试：`B=64,S=64,H=512,heads=4`。

| 路径 | torch_npu | custom | speedup | 精度 |
|---|---:|---:|---:|---:|
| sLSTM cell | `52.349 ms` | `3.658 ms` | `14.31x` | max diff `1.45e-06` |
| sLSTM layer | `53.910 ms` | `5.153 ms` | `10.46x` | 同一 cell 输出路径 |

两条性能路径都在 NPU 上执行，不是 CPU 性能对照。

## TorchAir fullgraph 补充验证

主测试 `B=64,S=64,H=512,heads=4` 的原表达被捕获为 2,376 个 FX 节点，随后在 `torch.ops.aten.log_sigmoid_forward.default` 处失败。

将 logsigmoid 人工等价改写为稳定的基础张量操作后：

| FX 节点 | 首次编译+运行 | 改写 eager | TorchAir | custom | 图/eager | custom/图 | max diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,888 | 158.61 s | 57.892 ms | 6.318 ms | 3.542 ms | 9.16x | 1.78x | `7.45e-08` |

人工改写需要修改模型代码，不能视为 TorchAir 自动优化；这反映当前 logsigmoid converter 覆盖不足。custom 直接实现目标 cell 语义，只需替换为 ACLNN 调用，不需要维护该 PyTorch 等价改写。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
