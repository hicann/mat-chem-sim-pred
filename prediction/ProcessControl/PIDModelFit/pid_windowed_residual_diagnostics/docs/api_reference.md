# PidWindowedResidualDiagnostics API Reference

## C API

```cpp
extern "C" int32_t aclnnPidWindowedResidualDiagnostics(
    void* actual,
    void* predicted,
    void* metrics,
    void* autocorr,
    int64_t batch,
    int64_t sample_count,
    int64_t window_size,
    int64_t stride,
    int64_t max_lag,
    void* workspace,
    uint64_t workspace_size,
    void* stream);

extern "C" uint64_t aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(
    int64_t batch,
    int64_t sample_count,
    int64_t window_size,
    int64_t stride,
    int64_t max_lag);

extern "C" int64_t aclnnPidWindowedResidualDiagnosticsGetWindowCount(
    int64_t sample_count,
    int64_t window_size,
    int64_t stride);
```

## Inputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `actual` | `float32` | `[batch, sample_count]` | Measured process output. |
| `predicted` | `float32` | `[batch, sample_count]` | Model-predicted process output. |

## Outputs

| Name | Type | Shape | Description |
|---|---|---|---|
| `metrics` | `float32` | `[batch, window_count, 8]` | Window-level residual diagnostics. |
| `autocorr` | `float32` | `[batch, window_count, max_lag]` | Residual autocorrelation for lag `1..max_lag`. |

The metric order is:

```text
0 mean_residual
1 std_residual
2 mae
3 rmse
4 max_abs_residual
5 fit_percent
6 durbin_watson
7 ljung_box_q
```

## Parameters

| Name | Constraint |
|---|---|
| `batch` | `> 0` |
| `sample_count` | `> 1` |
| `window_size` | `> 1` and `<= sample_count` |
| `stride` | `> 0` |
| `max_lag` | `> 0` and `< window_size` |
| `window_count` | `1 + (sample_count - window_size) / stride` |

## Workspace

Call `aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(...)` first and allocate a device workspace of at least that size. The current prototype only stores tiling metadata in workspace.

## Return Value

- `ACL_SUCCESS`: launch accepted.
- `ACL_ERROR_INVALID_PARAM`: null pointer, invalid shape, invalid workspace, or invalid stream.

The call launches asynchronously on `stream`; call `aclrtSynchronizeStream(stream)` before reading outputs.

## Example

```cpp
const int64_t window_count =
    aclnnPidWindowedResidualDiagnosticsGetWindowCount(sample_count, window_size, stride);

const uint64_t workspace_size = aclnnPidWindowedResidualDiagnosticsGetWorkspaceSize(
    batch, sample_count, window_size, stride, max_lag);

aclnnPidWindowedResidualDiagnostics(
    d_actual,
    d_predicted,
    d_metrics,
    d_autocorr,
    batch,
    sample_count,
    window_size,
    stride,
    max_lag,
    workspace,
    workspace_size,
    stream);
```

## Build And Run

```bash
cd prediction/ProcessControl/PIDModelFit/pid_windowed_residual_diagnostics
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DSOC_VERSION=Ascend910B3
cmake --build build -j 2
./build/test_aclnn_pid_windowed_residual_diagnostics 0
./build/benchmark_pid_windowed_residual_diagnostics 0 128 4096 512 256 32 5 64
```
