# {算子名称} — 算子示例模板

> 本模板说明如何在仓库中添加一个新的 Ascend C 算子。请参考已有算子（如 Lennard_Jones/、GAFF2/）作为完整实现示例。

---

## 目录结构

新建算子应遵循以下目录结构（`{算子名}` 使用小写+下划线命名）：

```
{算子名}/
├── CMakeLists.txt              # 顶层构建文件
├── README.md                   # 算子说明
├── docs/
│   ├── algorithm.md            # 算法说明（参见 template/algorithm.md）
│   └── api_reference.md        # API 参考（可选）
├── op_host/
│   ├── CMakeLists.txt          # Host 侧构建
│   ├── {算子名}.cpp             # Host 侧实现
│   └── {算子名}.h               # Host 侧头文件
├── op_kernel/
│   ├── CMakeLists.txt          # Kernel 侧构建
│   ├── {算子名}_kernel.cpp      # Kernel 实现
│   └── {算子名}_kernel.h        # Kernel 头文件
├── op_proto/
│   └── {算子名}_op.json         # 算子原型定义（可选）
├── examples/
│   ├── CMakeLists.txt
│   └── {算子名}_demo.cpp        # 使用示例
└── tests/
    ├── CMakeLists.txt           # 测试构建（递归包含子目录）
    ├── test_{算子名}.py          # Python 集成测试
    ├── benchmark_{算子名}.py     # 性能基准测试（可选）
    └── ut/
        ├── CMakeLists.txt
        └── op_kernel/
            ├── CMakeLists.txt
            └── test_{算子名}.cpp  # C++ 单元测试
```

## Host 侧代码结构

```cpp
// op_host/{算子名}.h
#pragma once

// 算子参数结构体
struct {算子名}Params {
    // 输入参数
    int32_t numAtoms;
    float paramA;
    float paramB;
    // Tiling 相关
    int32_t tileNum;
    uint64_t bufferSize;
};

// Host 侧调度类
class {算子名}Host {
public:
    static int32_t Apply(const {算子名}Params& params,
                         void* inputAddr,
                         void* outputAddr,
                         void* workspaceAddr,
                         uint64_t workspaceSize,
                         void* stream);
    static uint64_t GetWorkspaceSize(const {算子名}Params& params);
};
```

## Kernel 侧代码结构

```cpp
// op_kernel/{算子名}_kernel.h
#pragma once

// Kernel 入口函数
__aicore__ void {算子名}_kernel(void* input, void* output, void* workspace,
                                 {TiLingParams} tiling);
```

```cpp
// op_kernel/{算子名}_kernel.cpp
#include "{算子名}_kernel.h"

// 核心计算逻辑
__aicore__ void {算子名}_kernel(void* input, void* output, void* workspace,
                                 {TiLingParams} tiling) {
    // 1. 数据加载 (DataCopy)
    // 2. 计算处理 (向量化/标量化计算)
    // 3. 结果写回
}
```

## 使用示例

```cpp
// examples/{算子名}_demo.cpp
#include "acl/acl.h"
#include <iostream>

extern "C" int32_t aclnn{算子名}(
    void* inputAddr, void* outputAddr,
    int32_t paramA, float paramB,
    void* workspaceAddr, uint64_t workspaceSize, void* stream);

int main() {
    // 1. 初始化设备
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);

    // 2. 构造输入数据 & 分配设备内存
    // 3. 数据拷贝 Host → Device
    // 4. 调用算子
    // 5. 同步 & 取回结果
    // 6. 资源释放

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

## 构建配置

### 顶层 CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)
project({OPERATOR_NAME})

add_subdirectory(op_host)
add_subdirectory(op_kernel)
add_subdirectory(examples)
add_subdirectory(tests)
```

### op_host/CMakeLists.txt

```cmake
add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE {算子名} ACLNNTYPE aclnn_exclude)
```

> 更多构建配置请参考已有算子（如 `Lennard_Jones/`、`GAFF2/`）的实际 CMakeLists.txt 文件。