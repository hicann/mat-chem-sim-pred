# SelectiveScan1D API 说明

## Header

```cpp
#include "selective_scan_1d_host.h"
```

## Workspace 查询

```cpp
extern "C" uint64_t aclnnSelectiveScan1DGetWorkspaceSize(
    int64_t batch, int64_t length, int64_t dim, int64_t state);
```

返回值为 device workspace 字节数。当前 workspace 只保存 tiling 数据，并按 32 字节对齐。

## Host Launch

```cpp
extern "C" int32_t aclnnSelectiveScan1D(
    void* u,
    void* delta,
    void* a,
    void* b,
    void* c,
    void* d,
    void* output,
    int64_t batch,
    int64_t length,
    int64_t dim,
    int64_t state,
    void* workspace,
    uint64_t workspace_size,
    void* stream);
```

## 参数

| 参数 | 说明 |
|------|------|
| `u` | device pointer，float32 `[batch, length, dim]` |
| `delta` | device pointer，float32 `[batch, length, dim]` |
| `a` | device pointer，float32 `[dim, state]` |
| `b` | device pointer，float32 `[batch, length, state]` |
| `c` | device pointer，float32 `[batch, length, state]` |
| `d` | device pointer，float32 `[dim]` |
| `output` | device pointer，float32 `[batch, length, dim]` |
| `batch` | batch size，必须大于 0 |
| `length` | 序列长度，必须大于 0 |
| `dim` | scan channel 数，必须大于 0 |
| `state` | SSM state size，必须大于 0 |
| `workspace` | device workspace pointer，大小不小于查询结果 |
| `workspace_size` | workspace 字节数 |
| `stream` | `aclrtStream` |

## 返回值

| 返回值 | 含义 |
|--------|------|
| `ACL_SUCCESS` | 参数检查、tiling 拷贝和 kernel launch 成功提交 |
| `ACL_ERROR_INVALID_PARAM` | 空指针、非法 shape 或 workspace 不足 |
| 其他 ACL 错误码 | `aclrtMemcpyAsync` 等 runtime API 返回的错误 |

## Kernel launch

host 侧会把 tiling 写入 `workspace`，然后调用 kernel launcher：

```cpp
aclrtlaunch_selective_scan1_d(
    blockDim, stream, u, delta, a, b, c, d, output, nullptr, workspace);
```

`blockDim = min(32, batch * dim)`，至少为 1。

## 使用示例

```cpp
const uint64_t workspace_size =
    aclnnSelectiveScan1DGetWorkspaceSize(batch, length, dim, state);
void* workspace = nullptr;
aclrtMalloc(&workspace, workspace_size, ACL_MEM_MALLOC_HUGE_FIRST);

int32_t ret = aclnnSelectiveScan1D(
    d_u, d_delta, d_a, d_b, d_c, d_d, d_output,
    batch, length, dim, state, workspace, workspace_size, stream);
```

完整 ACL smoke 见 `examples/test_aclnn_selective_scan_1d.cpp`。

## 约束

- 当前实现只支持 float32。
- 输入/输出均为 contiguous ND layout。
- `workspace` 必须为 device memory。
- 调用方负责 `aclInit`、`aclrtSetDevice`、stream 创建与同步。
- host API 当前不持有 executor 对象；后续如改为标准 msopgen/aclnn executor 形态，应同步更新本文件和 smoke。
