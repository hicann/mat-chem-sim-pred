# PidFopdtBatchRolloutScore API

```cpp
extern "C" uint64_t aclnnPidFopdtBatchRolloutScoreGetWorkspaceSize(
    int64_t batch,
    int64_t candidates,
    int64_t sim_steps,
    int64_t candidate_tile);

extern "C" int32_t aclnnPidFopdtBatchRolloutScore(
    void* a,
    void* b,
    void* delay,
    void* y0,
    void* sp,
    void* kp,
    void* ki,
    void* kd,
    void* best_result,
    void* best_idx,
    int64_t batch,
    int64_t candidates,
    int64_t sim_steps,
    int64_t candidate_tile,
    float sample_interval,
    float settle_band,
    float overshoot_weight,
    float settling_weight,
    float control_weight,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## Output Layout

```text
best_score,best_kp,best_ki,best_kd,best_iae,best_ise,best_overshoot,best_settling_time
```

## Notes

- `candidate_tile` controls how many PID candidates are evaluated in one local-kernel launch. Pass `0` (the default) to let the host auto-select the optimal tile `min(candidates, kLane=768)`; `768` is the vector-lane throughput sweet spot, so callers normally leave it at `0`.
- `delay` is clamped to `0..31` inside the kernel
- workspace stores tile-local best results and the final reduction metadata
