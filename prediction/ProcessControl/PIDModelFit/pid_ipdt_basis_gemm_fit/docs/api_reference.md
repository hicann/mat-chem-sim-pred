# PidIpdtBasisGemmFit API

```cpp
extern "C" uint64_t aclnnPidIpdtBasisGemmFitGetWorkspaceSize(
    int64_t batch,
    int64_t candidates);

extern "C" int32_t aclnnPidIpdtBasisGemmFit(
    void* dot,
    void* basis_norm,
    void* y_energy,
    void* best_sse,
    void* best_k,
    void* best_idx,
    int64_t batch,
    int64_t candidates,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

`workspace_size` 不小于 `aclnnPidIpdtBasisGemmFitGetWorkspaceSize(batch, candidates)` 的返回值。
