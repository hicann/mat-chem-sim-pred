# PidStepResponseFeatures API Reference

## C API

```cpp
extern "C" int32_t aclnnPidStepResponseFeatures(
    void* pv_candidates,
    void* sp,
    void* features,
    int64_t batch,
    int64_t candidates,
    int64_t sample_count,
    float sample_interval,
    float settle_band_ratio,
    void* workspace,
    uint64_t workspace_size,
    void* stream);

extern "C" uint64_t aclnnPidStepResponseFeaturesGetWorkspaceSize(
    int64_t batch,
    int64_t candidates,
    int64_t sample_count);
```

## Inputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `pv_candidates` | `float32` | `[batch, candidates, sample_count]` | Candidate process-variable trajectories. |
| `sp` | `float32` | `[batch, sample_count]` | Setpoint trajectory for each loop. |

## Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `features` | `float32` | `[batch, candidates, 12]` | Step-response features for each candidate. |

Feature order:

```text
0  initial_value
1  final_value
2  final_abs_error
3  peak_value
4  trough_value
5  overshoot_ratio
6  undershoot_ratio
7  rise_time
8  peak_time
9  settling_time
10 iae
11 ise
```

## Parameters

| Name | Constraint |
|---|---|
| `batch` | `> 0` |
| `candidates` | `> 0` |
| `sample_count` | `> 1` |
| `sample_interval` | `> 0` |
| `settle_band_ratio` | `>= 0` |
| `batch * candidates` | must fit in `uint32_t` |

## Workspace

Call `aclnnPidStepResponseFeaturesGetWorkspaceSize(...)` first and allocate a device workspace of at least that size. The current prototype stores only tiling metadata in workspace.

## Return Value

- `ACL_SUCCESS`: launch accepted.
- `ACL_ERROR_INVALID_PARAM`: null pointer, invalid shape, invalid workspace, invalid stream, or invalid scalar parameter.

The call launches asynchronously on `stream`; call `aclrtSynchronizeStream(stream)` before reading outputs.

## Example

```cpp
const uint64_t workspace_size =
    aclnnPidStepResponseFeaturesGetWorkspaceSize(batch, candidates, sample_count);

aclnnPidStepResponseFeatures(
    d_pv_candidates,
    d_sp,
    d_features,
    batch,
    candidates,
    sample_count,
    sample_interval,
    settle_band_ratio,
    workspace,
    workspace_size,
    stream);
```
