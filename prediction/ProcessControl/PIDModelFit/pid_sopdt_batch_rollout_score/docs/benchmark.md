# PidSopdtBatchRolloutScore Benchmark Report

This document records the measured CPU/NPU behavior of `PidSopdtBatchRolloutScore`.

## Environment

- NPU host: `node202`
- Device: `Ascend910B3`, device id `0`
- CANN: `/usr/local/Ascend/ascend-toolkit/latest`
- CPU baseline: benchmark program multi-thread mode
- Build: `-DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3 -DRUN_MODE=npu`

## Method

The `benchmark_pid_sopdt_batch_rollout_score_aclnn` program builds an in-process multi-thread CPU reference
(`ComputeRange`, the same second-order recurrence `y[k+1] = a1*y[k] + a2*y[k-1] + b*u[k-delay]`), runs the
NPU operator on the same inputs and reports `max_abs_err`, `max_quality_rel_err` and `best_idx_diff_count`.
The pass conditions are `npu_zero_score_count == 0`, per-candidate scores matching the CPU reference to float32
precision, and any `best_idx` differences being near-ties (the chosen candidate's metric rel-err stays small).

The NPU state update is emitted in the same summation order as the CPU reference
(`(a1*y + a2*y_prev) + b*u`) so the long-horizon second-order recurrence stays float-aligned with it.

## Correctness

The SOPDT kernel differs from the verified FOPDT kernel only by adding one history state `y[k-1]` and one
coefficient. The candidate-axis SIMD width does not change the numerics (each tile is independent).

Measured on `node202 / Ascend910B3`, B=128, sim_steps=1024, candidate_tile=C, `npu_zero_score_count=0`:

| candidates | max_abs_err | max_quality_rel_err | best_idx_diff_count |
|---:|---:|---:|---:|
| 1024  | 2.4e-3 | 4.7e-5  | 0 |
| 4096  | 2.0    | 1.06e-2 | 6 |
| 16384 | 1.0    | 9.2e-3  | 14 |

At 1024 candidates the NPU output is essentially exact (rel-err `4.7e-5`). At 4096 the candidate grid samples
the same `Kp/Ki/Kd` range four times denser, so near the optimum many adjacent candidates have near-equal
scores; float-rounding then flips the arg-min for a few loops (`best_idx_diff_count=6`, growing to 14 at
16384 as the grid gets denser). This is a near-tie
effect, not a trajectory error: it persists at short horizons (e.g. `sim_steps=128` -> `best_idx_diff=6`,
`max_quality_rel_err=5.3e-3`), and the second-order oscillatory dynamics make SOPDT a bit more rounding-
sensitive than the first-order reference (FOPDT shows `max_quality_rel_err=4.5e-3, best_idx_diff=1` at 4096).
The `max_abs_err` (1-2) is again the discrete settling-time metric differing by 1-2 samples.

## Measured timing

`node202 / Ascend910B3`, B=128, sim_steps=1024, candidate_tile=C, CPU = 64-thread parallel reference.

| candidates | CPU parallel ms | NPU kernel ms | NPU kernel vs CPU |
|---:|---:|---:|---:|
| 1024  | 42.8  | 8.49  | 5.04x |
| 4096  | 143.8 | 28.4  | 5.07x |
| 16384 | 556.8 | 107.5 | 5.18x |

Against a 192-thread CPU reference the speedup is 4.1-4.4x (the wider CPU pool narrows the gap).

## Notes

The kernel reuses the FOPDT wide-lane (`kLane=768`) and fused inner-loop optimizations unchanged; the only
algorithmic difference is the second-order recurrence, which adds ~2 vector ops per timestep and one extra
state vector (placed in the previously-unused scratch block, so `kLane=768` still fits the 192 KB UB budget).
