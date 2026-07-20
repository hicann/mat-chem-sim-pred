# AutoCorrFusedAggregate Autoformer Forecasting E2E Report

Date: 2026-06-18
Device: Ascend 910B3

## Executive Summary

`AutoCorrFusedAggregate` now has the same Route B-style evidence chain used for `SelectiveScan1D`, at dataset-level forecasting scope.

The test compares two paths that differ only in the AutoCorrelation aggregation implementation:

- **Framework path**: torch_npu composition using lag loop + `torch.roll` + reduction + `topk` + softmax + gather/aggregate.
- **Custom path**: one custom Ascend C op, `AutoCorrFusedAggregate`.

All other work is shared torch_npu: CSV window tensorization, input projection, series decomposition, q/k/v projections, output projection, FFN, forecast head, MSE/MAE.

## Forecasting Task E2E Scope

Pipeline:

```text
MTO CSV -> normalization/windowing -> Autoformer-style block -> forecast head -> MSE/MAE
```

Dataset and shape:

| Item | Value |
|---|---:|
| rows | 4096 |
| numeric columns | 116 |
| selected features | 64 |
| eval windows | 8 |
| seq_len | 192 |
| pred_len | 24 |
| D_MODEL | 64 |
| heads x embed | 4 x 16 |
| top_k | 3 |
| repeat | 3 |
| custom ops per forward | 1 x `AutoCorrFusedAggregate` |

## Main Result: 8-window Forecasting E2E

| Path | Validation loop ms | Per-window ms | Throughput windows/s | MSE | MAE |
|---|---:|---:|---:|---:|---:|
| framework naive autocorr | 144.36 | 18.05 | 55.42 | 17.081244 | 2.911912 |
| custom AutoCorrFusedAggregate | 14.35 | 1.79 | 557.34 | 17.081244 | 2.911912 |

Comparison:

| Metric | Value |
|---|---:|
| validation-loop speedup | 10.06x |
| per-window speedup | 10.06x |
| AutoCorrelation aggregation speedup | 28.06x |
| MSE diff | 0.000e+00 |
| MAE diff | 0.000e+00 |
| prediction max abs diff | 5.960e-08 |
| prediction mean abs diff | 3.375e-09 |

## Component Timing

Per-window timing in ms:

| Component | Framework | Custom |
|---|---:|---:|
| input_projection | 0.1126 | 0.1022 |
| series_decomp | 0.3140 | 0.2574 |
| q_proj | 0.0915 | 0.0846 |
| k_proj | 0.0830 | 0.0756 |
| v_proj | 0.0800 | 0.0759 |
| autocorr_aggregate | 18.4998 | 0.6593 |
| out_proj | 0.1281 | 0.0896 |
| residual_norm | 0.2516 | 0.1893 |
| ffn | 0.1994 | 0.1709 |
| ffn_residual_norm | 0.1737 | 0.1606 |
| forecast_head | 0.1456 | 0.1263 |

AutoCorrelation is a real hotspot in the framework path:

- Framework AutoCorrelation share: **92.13%**
- Custom AutoCorrelation share: **33.10%**

## Route B Gate Conclusion

| Gate | Result | Evidence |
|---|---|---|
| Gate 1: framework weak/missing | PASS | torch_npu does not provide fused Autoformer AutoCorrelation; fallback is many small ops/lag loop. Screening measured 14.57-31.44 ms framework vs 0.288-0.455 ms custom at operator/subgraph level. |
| Gate 2: real hotspot | PASS | In forecasting E2E, framework AutoCorrelation aggregation is 92.13% of per-window block/task cost. |
| Gate 3: custom beats framework | PASS | Validation loop improves 10.06x; AutoCorrelation aggregation improves 28.06x; metrics are equivalent. |

## Reproduction

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /home/ql2025/work/tslib_cann_ops_dev
python3 autoformer_autocorr_e2e_ctypes.py   --windows 8   --repeat 3   --result-json docs/autocorr_autoformer_e2e_results.json
```

## Artifacts

- Script: `C:\tslib\autoformer_autocorr_e2e_ctypes.py`
- Result JSON: `C:\tslib\docs\autocorr_autoformer_e2e_results.json`
- This report: `C:\tslib\docs\autocorr_autoformer_e2e_report.md`
- Ascend 910B3 环境 mirror: `/home/ql2025/work/tslib_cann_ops_dev/`
