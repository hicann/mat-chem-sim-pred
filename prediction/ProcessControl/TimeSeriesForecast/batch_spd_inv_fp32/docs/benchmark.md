# BatchSpdInvFp32 性能报告

性能和 E2E 结果来自源交付物 C:/tslib/deliverables/batch_spd_inv_fp32_experimental。本次迁移保留了源侧结果文件：

- docs/source_batch_spd_inv_fp32_e2e_koopa.json
- docs/source_batch_spd_inv_fp32_results.json

迁移目录已在 Ascend 910B3 环境重新完成 msopgen 构建、ACLNN runtime 和正式 shape benchmark。

## 主要性能结果

| 场景 | 原框架路径 | 自定义路径 | speedup | 精度 |
|---|---:|---:|---:|---:|
| DMD, B16 m3 E128 | lstsq CPU fallback `932.47 ms` | 全设备 `0.386 ms` | `2418.55x` | max diff `5.96e-08` |
| DMD, B32 m7 E128 | lstsq CPU fallback `4341.75 ms` | 全设备 `0.339 ms` | `12823.56x` | max diff `6.80e-08` |
| Koopa E2E, seq336 pred96 B32 | fallback `5300.64 ms` | 全设备 `8.80 ms` | `602.19x` | max diff `2.38e-07` |

这里的原框架路径确实包含 CPU fallback，必须与其余算子的纯 NPU framework baseline 区分。

## TorchAir fullgraph 补充验证

正式 shape `B=32,m=7,E=128` 在运行时明确提示 `torch.linalg.lstsq` 不支持 NPU，并回退 CPU；单次中位时延为秒级。Dynamo 在 `aten.linalg_lstsq.default` 动态输出 shape 处失败，`fullgraph=True` 无法捕获，因此不存在可报告的 TorchAir NPU latency。

本算子按 CPU fallback 对比，不与其余 9 个算子的 NPU eager/TorchAir 口径混写。统一结论见 [第一批 TorchAir 报告](../../docs/torchair_first_batch_report.md)。
