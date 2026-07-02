# PidTuningRuleBatch API

```cpp
extern "C" uint64_t aclnnPidTuningRuleBatchGetWorkspaceSize(int64_t batch);

extern "C" int32_t aclnnPidTuningRuleBatch(
    void* process_gain,
    void* time_constant,
    void* dead_time,
    void* lambda_value,
    void* pid_params,
    void* diagnostics,
    int64_t batch,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## 参数

| 参数 | 说明 |
|------|------|
| `process_gain` | Device 输入，shape `[batch]`，FOPDT 过程增益 `K`，float32 |
| `time_constant` | Device 输入，shape `[batch]`，时间常数 `T`，float32 |
| `dead_time` | Device 输入，shape `[batch]`，纯滞后 `L`，float32 |
| `lambda_value` | Device 输入，shape `[batch]`，IMC 闭环时间常数 `lambda`，float32 |
| `pid_params` | Device 输出，shape `[batch, 3, 3]`，三类规则的 `(Kp, Ki, Kd)`，float32 |
| `diagnostics` | Device 输出，shape `[batch, 3, 4]`，`(valid, dead_time_ratio, aggressiveness, lambda_ratio)`，float32 |
| `batch` | 回路数 |
| `workspace` | Device workspace |
| `workspace_size` | 不小于 `aclnnPidTuningRuleBatchGetWorkspaceSize(batch)` |
| `stream` | ACL stream |

## 规则与参数顺序

```text
rule:  ziegler_nichols, imc, cohen_coon
param: kp, ki, kd
diag:  valid, dead_time_ratio, aggressiveness, lambda_ratio
```
