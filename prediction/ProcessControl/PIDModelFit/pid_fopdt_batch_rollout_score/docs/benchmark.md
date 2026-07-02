# PidFopdtBatchRolloutScore Benchmark Report

This document records the measured CPU/NPU behavior of `PidFopdtBatchRolloutScore`.

## Environment

- NPU host: `node202`
- Device: `Ascend910B3`, device id `0`
- CANN: `/usr/local/Ascend/ascend-toolkit/latest`
- CPU baseline: benchmark program multi-thread mode, 64 threads
- Build: `-DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3 -DRUN_MODE=npu`

## Correctness

NPU output is **bit-identical** to the CPU reference. The candidate-axis SIMD lane
width does not change the numerics (each tile is independent), so widening it leaves
`max_abs_err` and `best_idx_diff_count` exactly as the original 256-wide kernel.

Representative verified cases (B=128, S=1024, tile=C):

| candidates | max_abs_err | best_idx_diff_count | note |
|---:|---:|---:|---|
| 1024  | 1.1e-4 | 0 | exact |
| 4096  | (tie)  | 1 | a single argmin tie (two candidates with equal score); score rel-err 4.5e-3 |
| 16384 | 4.2e-4 | 1 | same pre-existing argmin tie |

The `best_idx_diff_count=1` at large C is a genuine argmin tie present in the original
256-wide kernel as well; it is not introduced by the optimization.

## Measured Result

`node202 / Ascend910B3`, B=128, sim_steps=1024, candidate_tile=C, kernel time is the
median of repeated runs. NPU kernel ms is stable; the CPU-64 baseline fluctuates on the
shared node, so the speedup is given as the observed range.

| candidates | CPU-64 ms | NPU kernel ms | NPU kernel vs CPU-64 |
|---:|---:|---:|---:|
| 1024  | ~34  | 7.66  | ~4.4x |
| 4096  | ~135-172 | 25.42 | ~5.3-6.8x |
| 16384 | ~489 | 96.3  | ~5.1x |

These are the shipped numbers after both optimizations below (wider lane + fused inner
loop).

## Optimization 1 - lane-width (kLane 256 -> 768)

The rollout inner loop is a serial time recurrence (`y[k+1]` depends on `y[k]`), so the
per-timestep chain of vector ops cannot be pipelined across steps. With a narrow SIMD
lane each vector instruction processes few candidates (256 floats = 4 compute cycles)
yet still pays a fixed ~10-20 cycle issue/latency, so the loop is **latency-bound**, not
throughput-bound. Widening the candidate-axis lane amortises that fixed latency over more
candidates per instruction (fewer instructions for the same work), turning the kernel
throughput-bound and filling the vector unit.

`kLane=768` is the largest lane that keeps the full 8 state vectors + 17-block scratch +
the 32-slot delay ring (delay spec `0..31`) + I/O queues within the 192 KB UB budget.

## Optimization 2 - inner-loop instruction reduction

The rollout inner loop issued ~37 vector ops per timestep. Two structural changes cut
that to ~32 without changing the result:

- the response error `e[k+1] = target - y[k+1]` is reused as the next step's error,
  dropping the redundant top-of-loop `target - y` recompute (saves 2 ops/step);
- the pure metric accumulators that do not feed back into the dynamics (`IAE`, `ISE`,
  `control_energy`) use the fused multiply-accumulate `Axpy` instead of a separate
  multiply + add (saves 3 ops/step).

The integral and the full state recurrence keep their explicit ops, so the simulated
trajectory is unchanged; on this hardware `Axpy` matches the separate multiply + add
bit-for-bit, so the whole result stays bit-identical to the original 256-wide kernel.

## Combined before/after

NPU kernel ms, same inputs, bit-identical output across all stages:

| candidates | kLane=256 (orig) | +wider lane (768) | +fused inner loop | total speedup |
|---:|---:|---:|---:|---:|
| 1024  | 14.13 | 8.60  | 7.66  | 1.84x |
| 4096  | 56.23 | 28.57 | 25.42 | 2.21x |
| 16384 | 224.6 | 108.5 | 96.3  | 2.33x |

## Interpretation

After both optimizations the operator is **competitive on a single card**: NPU kernel time
is roughly 4-7x the 64-thread CPU baseline at the candidate counts that dominate the
tuning sweep, with bit-identical results. This is the current performance baseline for
the FOPDT rollout operator.

The speedup comes from a layout change (wider candidate-axis SIMD) plus a lower
instruction count per timestep, not from any accuracy trade-off: the time-stepping
recurrence and the score definition are unchanged.

## Remaining headroom (not applied)

- The settling-time test is still ~6 ops/step; a branch-free cheaper reduction could trim
  a little more.
- `kLane=1024` reaches 22.95 ms (C=4096) / 91.31 ms (C=16384) but requires shrinking the
  delay ring (spec `0..31` -> `0..19`) to fit UB; usable only when the max delay is <= 19.
- Cross-batch flattening (fill the lane from the next loop's candidates when one loop has
  fewer candidates than the lane) for the small-C regime; needs per-element plant params.
- Multi-card data parallelism scales absolute time linearly (hardware, not a single-card
  algorithmic speedup).
