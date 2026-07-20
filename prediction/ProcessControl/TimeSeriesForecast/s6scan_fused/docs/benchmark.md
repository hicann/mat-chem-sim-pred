# S6scanFused 性能报告

## 环境

```text
Ascend 910B3 / CANN 8.1.RC1 / torch_npu 2.5.1
dtype: float32
baseline: torch_npu per-timestep loop baseline
```

## 结果

| 项目 | speedup |
|------|---------|
| component, single layer | `136.56x` |
| E2E, 3-layer encoder | `44.15x` |

历史结果来自 `C:\tslib\deliverables\s6scan_fused_experimental` 的 README / result JSON；迁移目录已在 Ascend 910B3 环境重新完成 msopgen 构建、ACLNN runtime 和正式 shape benchmark。

## TorchAir fullgraph 补充验证

主测试 `B=32,L=336,IN=11,H=64`：原 `softplus` 表达被 Dynamo 捕获为 5,053 个 FX 节点，但 TorchAir 缺少 `torch.ops.aten.softplus.default` converter，无法直接完整成图。

将 softplus 人工等价改写为 `maximum + abs + exp + log1p` 后：

| FX 节点 | 首次编译+运行 | 改写 eager | TorchAir | custom | 图/eager | custom/图 | max diff |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 7,069 | 141.83 s | 107.206 ms | 4.372 ms | 0.674 ms | 24.52x | 6.48x | `4.47e-08` |

人工改写需要修改模型代码，不能视为 TorchAir 自动优化；这反映当前 softplus converter 覆盖不足。custom 直接实现目标递推语义，只需替换为 ACLNN 调用，不需要维护该 PyTorch 等价改写。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
