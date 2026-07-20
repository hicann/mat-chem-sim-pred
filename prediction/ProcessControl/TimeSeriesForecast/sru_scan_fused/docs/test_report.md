# SruScanFused 测试报告

## 已完成

来自 `C:\tslib\deliverables\sru_scan_fused_experimental` 的 Ascend 910B3 环境 验证：

| 项目 | 结果 |
|------|------|
| msopgen JSON | 已提供 |
| Ascend C kernel | 已提供 |
| C++ probe vs CPU oracle | PASS |
| Python E2E vs torch_npu encoder | PASS |
| component speedup | `163.48x` |
| E2E speedup | `71.46x` |
| E2E max diff | `1.04e-07` |

## 本次迁移

已迁入或补齐：

- `op_kernel/sru_scan_fused_kernel.cpp`
- `op_host/sru_scan_fused_host.cpp`
- `op_host/sru_scan_fused_tiling.h`
- `msopgen/sru_scan_fused_msopgen.json`
- `examples/test_aclnn_sru_scan_fused.cpp`
- `tests/sru_scan_fused_probe.cpp`
- `tests/benchmark_sru_scan_fused_aclnn.cpp`
- `tests/sru_e2e.py`

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark：

```bash
cd prediction/ProcessControl/TimeSeriesForecast/sru_scan_fused
bash tests/run_probe.sh
bash tests/run_e2e.sh
```

## TorchAir fullgraph 验证

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B32,L336,IN11,H64` |
| 完整成图 | PASS，7,069 FX 节点 |
| 首次编译+运行 | `165.43 s` |
| eager / TorchAir / custom | `118.829 / 5.272 / 0.884 ms` |
| custom vs TorchAir | `5.97x` |
| graph/custom max diff | `0 / 1.79e-07` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
