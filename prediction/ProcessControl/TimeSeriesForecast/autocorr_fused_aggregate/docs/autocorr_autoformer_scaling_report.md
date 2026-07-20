# AutoCorrFusedAggregate Autoformer Forecasting Scaling Report

Date: 2026-06-18
Device: Ascend 910B3

## Question

As with `SelectiveScan1D`, does the advantage become more obvious as the dataset/window count grows?

Answer: **yes, but in the same amortization sense**. The steady-state model loop stays around **10-16x** faster, while all-in speedup including CSV load/preprocess grows from **1.10x** at 1 window to **1.73x** at 16 windows.

## Fixed Setup

- Same script/model/data as `autocorr_autoformer_e2e_report.md`
- Changed variable: validation windows = 1/2/4/8/16
- Repeat per point: 2
- Custom ops used: only `AutoCorrFusedAggregate`

## Results

| Windows | Framework loop ms | Custom loop ms | Steady loop speedup | All-in speedup incl. CSV | Custom throughput windows/s | Pred max diff |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 34.41 | 2.15 | 16.01x | 1.10x | 465.38 | 5.96e-08 |
| 2 | 47.36 | 3.54 | 13.37x | 1.16x | 564.66 | 5.96e-08 |
| 4 | 82.77 | 7.08 | 11.68x | 1.24x | 564.68 | 5.96e-08 |
| 8 | 151.28 | 14.36 | 10.54x | 1.41x | 557.17 | 5.96e-08 |
| 16 | 279.38 | 27.49 | 10.16x | 1.73x | 582.13 | 8.94e-08 |

## Interpretation

1. **Steady-state model speedup is strong**: custom stays around 465-582 windows/s while framework stays around 29-57 windows/s.
2. **The all-in number is lower for small window counts** because CSV parsing/normalization is identical fixed overhead.
3. **More windows amortize the fixed cost**, so all-in speedup increases with window count.
4. **Numerics remain equivalent**: MSE/MAE diffs are zero in all points; prediction max diff stays below `1e-7`.

## Artifacts

- Scaling JSON: `C:\tslib\docs\autocorr_autoformer_scaling_results.json`
- This report: `C:\tslib\docs\autocorr_autoformer_scaling_report.md`
- Ascend 910B3 环境 mirror: `/home/ql2025/work/tslib_cann_ops_dev/docs/`
