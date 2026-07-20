# LtcScanFused 测试报告

## 已完成

来自 `C:\tslib\deliverables\ltc_scan_fused_experimental` 的 Ascend 910B3 环境 验证：

| 项目 | 结果 |
|------|------|
| msopgen JSON | 已提供 |
| Ascend C kernel | 已提供 |
| op_host tiling / InferShape / OpDef | 已提供 |
| C++ probe vs CPU oracle | PASS |
| Python E2E vs torch_npu encoder | PASS |
| component speedup | `82.44x` |
| E2E speedup | `76.90x` |
| E2E max diff | `2.53e-07` |

## 本次迁移

已迁入：

- `op_kernel/ltc_scan_fused_kernel.cpp`
- `op_host/ltc_scan_fused_host.cpp`
- `op_host/ltc_scan_fused_tiling.h`
- `msopgen/ltc_scan_fused_msopgen.json`
- `examples/test_aclnn_ltc_scan_fused.cpp`
- `tests/ltc_scan_fused_probe.cpp`
- `tests/benchmark_ltc_scan_fused_aclnn.cpp`
- `tests/ltc_e2e.py`

## Ascend 910B3 复验

迁移目录已完成 msopgen 生成、C++ 构建、ACLNN runtime、正确性和正式 shape benchmark：

```bash
cd prediction/ProcessControl/TimeSeriesForecast/ltc_scan_fused
bash tests/run_probe.sh
bash tests/run_e2e.sh
```

## TorchAir fullgraph 验证

| 项目 | 结果 |
|---|---|
| 主测试 shape | `B32,L336,IN11,H64,K6` |
| 完整成图 | PASS，24,878 FX 节点 |
| 首次编译+运行 | `616.61 s` |
| eager / TorchAir / custom | `386.003 / 10.227 / 4.909 ms` |
| custom vs TorchAir | `2.08x` |
| graph/custom max diff | `1.34e-07 / 1.34e-07` |

详情见 [benchmark.md](benchmark.md) 和 [第一批总览](../../docs/torchair_first_batch_report.md)。
