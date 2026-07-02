# PidResidualDiagnostics API

```cpp
extern "C" uint64_t aclnnPidResidualDiagnosticsGetWorkspaceSize(
    int64_t batch,
    int64_t sample_count,
    int64_t max_lag);

extern "C" int32_t aclnnPidResidualDiagnostics(
    void* actual,
    void* predicted,
    void* metrics,
    void* autocorr,
    int64_t batch,
    int64_t sample_count,
    int64_t max_lag,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## 参数

| 参数 | 说明 |
|------|------|
| `actual` | Device 输入，shape `[batch, sample_count]`，float32，实测过程输出 |
| `predicted` | Device 输入，shape `[batch, sample_count]`，float32，模型预测输出 |
| `metrics` | Device 输出，shape `[batch, 8]`，float32 |
| `autocorr` | Device 输出，shape `[batch, max_lag]`，float32 |
| `batch` | 回路数量 |
| `sample_count` | 每条回路采样点数，必须大于 1 |
| `max_lag` | 自相关最大滞后阶数，范围为 `1 <= max_lag < sample_count` |
| `workspace` | Device workspace |
| `workspace_size` | workspace 大小，必须不小于 `aclnnPidResidualDiagnosticsGetWorkspaceSize` 返回值 |
| `stream` | ACL stream |

## 输出指标顺序

```text
mean_residual,std_residual,mae,rmse,max_abs_residual,fit_percent,durbin_watson,ljung_box_q
```

`autocorr[b, lag - 1]` 对应第 `lag` 阶残差自相关，`lag` 从 1 到 `max_lag`。

## 返回值

| 返回值 | 说明 |
|--------|------|
| `ACL_SUCCESS` | launch 成功 |
| `ACL_ERROR_INVALID_PARAM` | 输入指针为空、shape 参数非法或 workspace 不足 |

## 约束

- 当前原型仅支持 float32。
- 输入和输出均为连续 ND 布局。
- `actual/predicted/metrics/autocorr/workspace` 必须位于 Device 内存。
- `max_lag` 越大，计算量按 `B * N * max_lag` 增长。
