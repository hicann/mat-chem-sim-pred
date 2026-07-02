# PidIpdtBatchRolloutScore Algorithm

## Purpose

This operator evaluates many PID candidates for many IPDT loops during the tuning stage and returns the best
candidate for each loop.

The target workload is:

```text
batch loops x candidate set x rollout time steps
```

## Model

The plant model is discretized IPDT (integrator plus dead time):

```text
y[k+1] = y[k] + b * u[k-delay]
```

This is the `a = 1` special case of the FOPDT recurrence `y[k+1] = a*y[k] + b*u[k-delay]`: there is no
self-regulating decay term, so the plant integrates the (delayed) control input. There is no `a` input;
`b` is the per-step integration gain. Everything else (PID law, scoring, candidate-axis SIMD, delay ring,
tiling) is identical to `PidFopdtBatchRolloutScore`.

The PID law is:

```text
e[k] = sp - y[k]
integral += e[k] * dt
derivative = (e[k] - e[k-1]) / dt
u[k] = clamp(Kp * e[k] + Ki * integral + Kd * derivative, -10, 10)
```

## Score

For each candidate, the rollout accumulates:

- `IAE`
- `ISE`
- `overshoot`
- `settling_time`
- `control_energy`

The optimization target is:

```text
score = IAE
      + overshoot_weight * overshoot
      + settling_weight * settling_time
      + control_weight * control_energy
```

The operator returns the candidate with minimum `score`.

## NPU Execution Strategy

The current implementation uses a two-stage tiled structure:

1. host splits the candidate axis into tiles
2. local kernel evaluates one tile for all assigned loops and writes partial best results
3. final kernel reduces all tile-local best results into one best result per loop

This structure was chosen because the earlier single-launch `(loop, tile)` task mapping showed unstable coverage
on `node202`. The current host-per-tile launch plus conservative loop-range partitioning restores correctness.

## Kernel difference from FOPDT

The only change versus the FOPDT kernel is the state recurrence: the `Muls(ytmp, y, model_a)` decay step is
dropped and `y` is updated in place as `Add(y, y, model_b * u[k-delay])`. The `a` GM input and its tensor are
removed. The state vector count, scratch layout, delay ring and reduction are unchanged, so `kLane=768`
still fits the 192 KB UB budget.

## Vectorization

The rollout time dimension is a serial recurrence (`y[k+1]` depends on `y[k]`) and cannot be turned into
GEMM-style dense math without dropping the per-step nonlinearities (control clamp) and the nonlinear score
functionals (IAE/ISE/overshoot/settling), so the kernel keeps the exact step-by-step recurrence.

The parallelism instead lives on the candidate axis: every timestep applies the same chain of vector ops to
all candidates at once. Because the recurrence is serial, that chain of dependent vector ops cannot be
pipelined across timesteps, so with a narrow lane the inner loop is bound by per-instruction issue/latency
rather than by compute throughput. The kernel therefore evaluates the candidate axis with a wide SIMD lane
(`kLane=768`): more candidates per vector instruction means fewer instructions for the same work, which
amortises the fixed instruction latency and makes the loop throughput-bound. `kLane=768` is the largest lane
that keeps the 8 state vectors + scratch + the 32-slot delay ring (delay spec `0..31`) + I/O queues within
the 192 KB UB budget. Widening the lane is a pure layout change and leaves the output bit-identical.

## Engineering Conclusion

This operator is valuable as:

- an independent PID tuning operator sample
- a correctness-verified NPU exploration artifact (NPU output matches the CPU reference, quality rel-err `< 1e-3`)
- a single-card rollout that reuses the FOPDT wide-lane plus fused inner-loop optimizations

The inner loop was also reduced from ~37 to ~32 vector ops per timestep by reusing the response error as the
next step's error and by folding the non-feedback metric accumulators (IAE/ISE/control energy) into fused
multiply-accumulates; this is bit-identical to the original. The remaining single-card headroom is a cheaper
settling reduction; multi-card data parallelism scales the absolute time further but is a hardware lever, not
a single-card algorithmic speedup.
