# PidProcessCapabilityMetrics API

```cpp
extern "C" uint64_t aclnnPidProcessCapabilityMetricsGetWorkspaceSize(
    int64_t batch,
    int64_t sample_count);

extern "C" int32_t aclnnPidProcessCapabilityMetrics(
    void* values,
    void* lsl,
    void* usl,
    void* metrics,
    int64_t batch,
    int64_t sample_count,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## 参数

| 参数 | 说明 |
|------|------|
| `values` | Device 输入，shape `[batch, sample_count]`，float32 |
| `lsl` | Device 输入，shape `[batch]`，float32 |
| `usl` | Device 输入，shape `[batch]`，float32 |
| `metrics` | Device 输出，shape `[batch, 13]`，float32 |
| `batch` | 回路数量 |
| `sample_count` | 每条回路窗口采样点数 |
| `workspace` | Device workspace |
| `workspace_size` | workspace 大小 |
| `stream` | ACL stream |

## 输出指标顺序

```text
mean,std_sample,std_population,cp,cpu,cpl,cpk,pp,ppk,out_ratio,out_count,min,max
```

