# PidControlPerformanceMetrics API

```cpp
extern "C" uint64_t aclnnPidControlPerformanceMetricsGetWorkspaceSize(
    int64_t batch,
    int64_t sample_count);

extern "C" int32_t aclnnPidControlPerformanceMetrics(
    void* pv,
    void* sp,
    void* lsl,
    void* usl,
    void* mv_variance,
    void* metrics,
    int64_t batch,
    int64_t sample_count,
    float sample_interval,
    float settle_band,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## 参数

| 参数 | 说明 |
|------|------|
| `pv` | Device 输入，shape `[batch, sample_count]`，float32 |
| `sp` | Device 输入，shape `[batch, sample_count]`，float32 |
| `lsl` | Device 输入，shape `[batch]`，float32 |
| `usl` | Device 输入，shape `[batch]`，float32 |
| `mv_variance` | Device 输入，shape `[batch]`，float32 |
| `metrics` | Device 输出，shape `[batch, 20]`，float32 |
| `batch` | 回路数量 |
| `sample_count` | 每条回路窗口采样点数 |
| `sample_interval` | 采样间隔，用于 IAE/ISE/ITAE/settling_time |
| `settle_band` | 稳定带，`abs(sp - pv) <= settle_band` 视为稳定 |
| `workspace` | Device workspace |
| `workspace_size` | workspace 大小 |
| `stream` | ACL stream |

## 输出指标顺序

```text
mean_pv,std_pv_sample,std_pv_population,cp,cpk,pp,ppk,harris_index,iae,ise,
itae,mae,rmse,max_abs_error,out_ratio,out_count,overshoot,undershoot,settling_time,final_abs_error
```
