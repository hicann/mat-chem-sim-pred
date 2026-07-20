# AutoCorrFusedAggregate - fused Autoformer autocorrelation aggregation

`AutoCorrFusedAggregate` fuses Autoformer-style autocorrelation lag scoring, top-k lag selection, softmax weighting, and circular value aggregation into one Ascend C kernel.

## Op

```text
AutoCorrFusedAggregate(query:[B,H,E,L], key:[B,H,E,L], value:[B,H,E,L], top_k:int=1) -> output:[B,H,E,L]
```

For each independent `(batch, head, embed)` sequence, the kernel:

1. computes circular autocorrelation scores for every lag;
2. keeps the top-k lags;
3. applies softmax over the selected scores;
4. aggregates circularly shifted `value` sequences.

## Three-gate results

| Gate | Result |
|------|--------|
| Framework weak/missing | PASS: torch_npu path is a lag loop with `roll`, `topk`, softmax, gather, and reduction |
| Real hotspot | PASS: framework AutoCorrelation aggregation is 92.13% of the Autoformer-style per-window block cost |
| Custom beats framework | PASS: validation loop `144.36 ms -> 14.35 ms`, speedup `10.06x`; aggregation speedup `28.06x`; prediction max diff `5.96e-08` |

## Files

- `op_kernel/autocorr_fused_aggregate_kernel.cpp` - fused Ascend C kernel
- `op_host/autocorr_fused_aggregate_host.cpp` - op-host tiling, shape inference, and OpDef
- `op_host/auto_corr_fused_aggregate_tiling.h` - tiling data definition
- `msopgen/autocorr_fused_aggregate_msopgen.json` - msopgen IR
- `examples/test_aclnn_autocorr_fused_aggregate.cpp` - ACLNN smoke example
- `tests/benchmark_autocorr_fused_aggregate_aclnn.cpp` - component benchmark
- `tests/autoformer_autocorr_e2e_ctypes.py` - Autoformer-style forecasting E2E script
- `docs/autocorr_autoformer_e2e_report.md` and `docs/autocorr_autoformer_scaling_report.md` - historical Ascend 910B3 reports

## Build

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
msopgen gen -i msopgen/autocorr_fused_aggregate_msopgen.json -f aclnn -c ai_core-ascend910b -out build/msopgen_autocorr_fused_aggregate -lan cpp
```

After generation, replace the generated kernel/host stubs with this directory's `op_kernel` and `op_host` files.

## TorchAir 补充结论

原模型 `roll` 表达在主测试 `B1,heads4,embed16,L192,top_k3` 上无法直接完整成图，失败原因为缺少 `aten.roll.default` converter。将循环移位人工等价改写为静态 `slice + cat` 后，TorchAir 为 `4.568 ms`，custom 为 `0.618 ms`，custom 仍快 `7.39x`；首次编译和运行耗时 `69.70 s`。人工改写结果是强框架基线，不代表 TorchAir 能自动优化原模型。custom 直接实现 lag 聚合语义，接入时只需替换为 ACLNN 调用，不需要为了缺失 converter 重写 `roll` 表达。

改进后的指数近似在模型量级和宽动态范围输入上的 max diff 分别为 `7.45e-09` 和 `1.43e-05`。最终正负指数缩放分支重建后复验为 `0.610/0.633 ms`，误差保持不变。详情见 [docs/benchmark.md](docs/benchmark.md) 和 [第一批总览](../docs/torchair_first_batch_report.md)。
