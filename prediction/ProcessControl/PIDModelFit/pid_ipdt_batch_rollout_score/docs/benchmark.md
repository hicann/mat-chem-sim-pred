# PidIpdtBatchRolloutScore Benchmark Report

This document records the measured CPU/NPU behavior of `PidIpdtBatchRolloutScore`.

## Environment

- NPU host: `node202`
- Device: `Ascend910B3`, device id `0`
- CANN: `/usr/local/Ascend/ascend-toolkit/latest`
- CPU baseline: benchmark program multi-thread mode
- Build: `-DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3 -DRUN_MODE=npu`

## Method

The `benchmark_pid_ipdt_batch_rollout_score_aclnn` program builds an in-process multi-thread CPU reference
(`ComputeRange`, the same integrator recurrence `y[k+1] = y[k] + b*u[k-delay]`), runs the NPU operator on the
same inputs and reports `max_abs_err`, `max_quality_rel_err` and `best_idx_diff_count`. The pass conditions
are `npu_zero_score_count == 0`, per-candidate scores matching the CPU reference to float32 precision, and any
`best_idx` differences being near-ties (the chosen candidate's metric rel-err stays small), matching the
behavior of the verified FOPDT operator.

## Correctness

The IPDT kernel differs from the verified FOPDT kernel only in the state recurrence (the `a*y` decay term is
dropped). The candidate-axis SIMD width does not change the numerics (each tile is independent), so the
accuracy profile matches FOPDT: NPU output equals the CPU reference within float32 rounding.

Measured on `node202 / Ascend910B3`, B=128, sim_steps=1024, candidate_tile=C, `npu_zero_score_count=0`:

| candidates | max_abs_err | max_quality_rel_err | best_idx_diff_count |
|---:|---:|---:|---:|
| 1024  | 2.4e-4 | 1.5e-6  | 0 |
| 4096  | 1.0    | 1.69e-3 | 1 |
| 16384 | 1.5e-3 | 3.3e-5  | 1 |

The `max_abs_err=1` at 4096 is the discrete settling-time metric crossing the settle band one sample later
on NPU than on CPU for a single near-tie loop (`dt=1` -> abs diff 1); the corresponding metric rel-err stays
`< 2e-3`. The reference FOPDT operator shows the same behavior at this candidate count
(`max_abs_err=1, max_quality_rel_err=4.5e-3, best_idx_diff_count=1`), so IPDT is within the accepted baseline.

## Measured timing

`node202 / Ascend910B3`, B=128, sim_steps=1024, candidate_tile=C, CPU = 64-thread parallel reference.

| candidates | CPU parallel ms | NPU kernel ms | NPU kernel vs CPU |
|---:|---:|---:|---:|
| 1024  | 32.5  | 7.45 | 4.36x |
| 4096  | 122.1 | 24.7 | 4.95x |
| 16384 | 426.6 | 93.8 | 4.55x |

Against a 192-thread CPU reference the speedup is 3.8-4.0x (the wider CPU pool narrows the gap).

## Notes

The kernel reuses the FOPDT wide-lane (`kLane=768`) and fused inner-loop optimizations unchanged; the only
algorithmic difference is the integrator recurrence, which removes one vector multiply per timestep.
